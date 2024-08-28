from collections import OrderedDict
import math
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.utils import (
    logging,
)
from lpips import LPIPS
from functools import partial

from .masked_lpips import MaskedLPIPS
from .cross_attention import (prepare_unet, free_unet, get_and_release_last_attention,
                              get_initial_cross_attention_all_timesteps, p2p_reshape_initial_cross_attention,
                              get_attention_interpolation_factor, XA_STORE_INITIAL_CONDITIONAL_ONLY,
                              apply_segmentation_postprocessing)

logger = logging.get_logger(__name__)


class DenoisingLoop(torch.nn.Module):
    def __init__(self, unet, scheduler, progress_bar, num_inference_steps, timesteps, do_classifier_free_guidance,
                 guidance_scale, loss_steps, per_timestep_null_text=False, per_timestep_conditioning_delta=False,
                 per_timestep_uncond_delta=False, xa_interpolation_schedule=None, self_interpolation_schedule=None,
                 do_checkpointing=True, solver_order=1):
        super().__init__()
        self.scheduler = scheduler
        self.progress_bar = progress_bar
        self.per_timestep_conditioning_delta = per_timestep_conditioning_delta
        self.per_timestep_uncond_delta = per_timestep_uncond_delta
        self.per_timestep_null_text = per_timestep_null_text
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.guidance_scale = guidance_scale
        self.do_checkpointing = do_checkpointing
        self.solver_order = solver_order
        self.unet = unet
        self.xa_interpolation_schedule = xa_interpolation_schedule
        self.self_interpolation_schedule = self_interpolation_schedule
        self.loss_steps = loss_steps

    def get_step_function(self, collect_xa):
        def step_function(latents, uncond_embeds, uncond_embeds_delta, conditional_embeds, conditional_embeds_delta,
                          i, t, extra_step_kwargs):
            per_timestep_idx = i // self.solver_order + i % self.solver_order
            if conditional_embeds_delta is not None:
                if self.per_timestep_conditioning_delta:
                    conditional_embeds_delta_i = conditional_embeds_delta[per_timestep_idx]
                else:
                    conditional_embeds_delta_i = conditional_embeds_delta
                cond_i = conditional_embeds + conditional_embeds_delta_i
            else:
                cond_i = conditional_embeds

            if self.per_timestep_null_text:
                uncond_i = uncond_embeds[per_timestep_idx]
            else:
                uncond_i = uncond_embeds

            if uncond_embeds_delta is not None:
                if self.per_timestep_uncond_delta:
                    uncond_embeds_delta_i = uncond_embeds_delta[per_timestep_idx]
                else:
                    uncond_embeds_delta_i = uncond_embeds_delta
                uncond_i = uncond_i + uncond_embeds_delta_i

            prompt_embeds = torch.cat([uncond_i, cond_i])
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            xa_attention_initial_interpolation_factor = \
                get_attention_interpolation_factor(i, self.timesteps, self.xa_interpolation_schedule)
            self_attention_initial_interpolation_factor = \
                get_attention_interpolation_factor(i, self.timesteps, self.self_interpolation_schedule)
            xa_args = {"timestep": t,
                       "xa_attention_initial_interpolation_factor": xa_attention_initial_interpolation_factor,
                       "self_attention_initial_interpolation_factor": self_attention_initial_interpolation_factor}
            unet_out = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds,
                                 cross_attention_kwargs=xa_args)
            noise_pred = unet_out.sample

            xa_maps_t = None
            xa_map_names = None
            if collect_xa:
                xa_maps_t, xa_map_names = get_and_release_last_attention(self.unet, i)

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            pred_original_sample = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample

            if collect_xa:
                return latents, pred_original_sample, xa_maps_t, xa_map_names
            else:
                return latents, pred_original_sample

        return step_function

    def __call__(self, uncond_embeds, conditional_embeds, uncond_embeds_delta, conditional_embeds_delta, latents,
                 extra_step_kwargs=None, collect_xa=False):
        self.scheduler.set_timesteps(self.num_inference_steps, device=latents.device)
        if collect_xa:
            xa_maps = {}
        else:
            xa_maps = None

        pred_originals = OrderedDict()
        for i, t in enumerate(self.timesteps):
            step_function = self.get_step_function(collect_xa)
            step_return = step_function(latents, uncond_embeds, uncond_embeds_delta,
                                        conditional_embeds, conditional_embeds_delta,
                                        i, t, extra_step_kwargs)
            if not collect_xa:
                latents, pred_original_sample = step_return
            else:
                latents, pred_original_sample, xa_maps_t, xa_map_names = step_return

                xa_maps_t = {}
                for xa_name, xa_map in zip(xa_map_names, xa_maps_t):
                    xa_maps_t[xa_name] = xa_map
                xa_maps[t.item()] = xa_maps_t

            if i >= (len(self.timesteps) - self.loss_steps):
                pred_originals[i] = pred_original_sample
            else:
                pred_originals[i] = None

        return pred_originals, xa_maps


def enable_gradient_checkpointing(model: torch.nn.Module):
    model.enable_gradient_checkpointing()
    model.train()
    module_different_train_behaviour = (torch.nn.modules.batchnorm._BatchNorm,
                                        torch.nn.modules.instancenorm._InstanceNorm,
                                        torch.nn.modules.dropout._DropoutNd,
                                        )
    for module_name, module in model.named_modules():
        if isinstance(module, module_different_train_behaviour):
            module.eval()
            #print(module_name)


class StableDiffusionPipelineWithGrad(StableDiffusionPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPFeatureExtractor,
                 requires_safety_checker: bool = True,
                 ):
        super().__init__(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
                         safety_checker=safety_checker, feature_extractor=feature_extractor,
                         requires_safety_checker=requires_safety_checker)
        self.dtype = torch.float32

        enable_gradient_checkpointing(self.unet)
        enable_gradient_checkpointing(self.vae)

    def decode_latent_to_img(self, latent_in):
        latents = 1 / self.vae.config.scaling_factor * latent_in
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5)
        image = torch.clamp(image, 0.0, 1.0)
        return image

    def encode_image_to_latent(self, img_in):
        img_rescaled = 2 * img_in[None,:,:,:] - 1
        latent = self.vae.encode(img_rescaled).latent_dist.mean
        latent = latent * self.vae.config.scaling_factor
        return latent

    def convert_to_double(self):
        self.dtype= torch.double
        self.unet.double()
        self.text_encoder.double()
        #stuff in image space is typically fine in single precision
        #self.vae.double()

    @torch.no_grad()
    def __call__(self, targets,
                 starting_img=None,
                 loss_function=None,
                 loss_w=1.0,
                 regularizers_weights={},
                 sgd_steps=20,
                 sgd_stepsize=0.05,
                 latent_lr_factor=1.0,
                 conditioning_lr_factor=1.0,
                 uncond_lr_factor=1.0,
                 early_stopping_loss=None,
                 sgd_optim='sgd',
                 sgd_scheduler=None,
                 loss_steps=1,
                 loss_steps_schedule='uniform',
                 solver_order=1,
                 optimize_latents=False,
                 optimize_conditioning=False,
                 per_timestep_conditioning_delta=False,
                 optimize_uncond=False,
                 per_timestep_uncond_delta=False,
                 normalize_gradients=False,
                 gradient_clipping=None,
                 null_text_embeddings=None,
                 xa_interpolation_schedule=None,
                 self_interpolation_schedule=None,
                 prompt: Union[str, List[str]] = None,
                 prompt_foreground_key_words: List[str] = None,
                 squash_function = None,
                 height: Optional[int] = None, width: Optional[int] = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 negative_prompt: Optional[Union[str, List[str]]] = None,
                 num_images_per_prompt: Optional[int] = 1,
                 eta: float = 0.0,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 latents: Optional[torch.FloatTensor] = None,
                 prompt_embeds: Optional[torch.FloatTensor] = None,
                 prompt_to_prompt_replacements=None,
                 negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                 output_type: Optional[str] = "pil",
                 return_dict: bool = True,
                 callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                 callback_steps: Optional[int] = 1,
                 ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        if not do_classifier_free_guidance:
            print('We only support optimization with guidance scale > 1.0')

        # 3. Encode input prompt
        assert prompt_embeds is None and negative_prompt_embeds is None

        with torch.no_grad():
            conditional_embeds, uncond_embeds, word_to_token_embeddings = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )

        #calculate foreground token mask for segmentation calculation
        foreground_token_mask = None
        if prompt_foreground_key_words is not None:
            foreground_token_mask = torch.zeros(conditional_embeds.shape[1], device=conditional_embeds.device, dtype=torch.bool)
            for word in prompt_foreground_key_words:
                for word_key, word_positions in word_to_token_embeddings.items():
                    if word == word_key:
                        for position in word_positions:
                            foreground_token_mask[position] = 1


        if prompt_to_prompt_replacements is not None:
            p2p_prompt = prompt

            #replace words and note the positions in the prompt
            source_word, target_word = prompt_to_prompt_replacements
            assert source_word in prompt
            p2p_prompt = p2p_prompt.replace(source_word, target_word)

            with torch.no_grad():
                p2p_conditional_embeds, p2p_uncond_embeds , p2p_word_to_token_embeddings = self.encode_prompt(
                    p2p_prompt,
                    device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                )

        #Setup
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        num_images_per_prompt = 1
        generator = None
        initial_latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            uncond_embeds.dtype,
            device,
            generator,
            latents,
        )
        initial_latents = initial_latents.to(self.dtype)
        initial_latents = initial_latents.detach()
        initial_latents_norm = torch.norm(initial_latents.view(-1), p=2)

        if optimize_conditioning:
            if per_timestep_conditioning_delta:
                conditioning_delta_shape = (num_inference_steps,) + conditional_embeds.shape
            else:
                conditioning_delta_shape = conditional_embeds.shape

            conditional_embeds_delta = torch.zeros(conditioning_delta_shape,
                                                   device=device, dtype=conditional_embeds.dtype)
        else:
            conditional_embeds_delta = None

        if optimize_uncond:
            if per_timestep_uncond_delta:
                uncond_delta_shape = (num_inference_steps,) + uncond_embeds.shape
            else:
                uncond_delta_shape = uncond_embeds.shape

            uncond_embeds_delta = torch.zeros(uncond_delta_shape,
                                                   device=device, dtype=uncond_embeds.dtype)
        else:
            uncond_embeds_delta = None

        if null_text_embeddings is not None:
            print('Using pre-defined Null-Text prompts')
            uncond_embeds = null_text_embeddings.to(device)
            per_timestep_null_text = True
        else:
            per_timestep_null_text = False

        #setup optimizer
        optim = self.create_optimizer(initial_latents, uncond_embeds_delta, conditional_embeds_delta, optimize_latents,
                                      optimize_uncond, optimize_conditioning, sgd_optim, sgd_stepsize, latent_lr_factor,
                                      conditioning_lr_factor, uncond_lr_factor)

        if sgd_scheduler is None or sgd_scheduler == 'none':
            optim_scheduler = None
        elif sgd_scheduler == 'cosine':
            optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, sgd_steps, eta_min=sgd_stepsize / 10)
        else:
            raise NotImplementedError()

        #DDIM Loop with gradient checkpointing
        ddim_loop = DenoisingLoop(self.unet, self.scheduler, self.progress_bar, num_inference_steps, timesteps,
                                  do_classifier_free_guidance, guidance_scale, loss_steps,
                                  per_timestep_null_text=per_timestep_null_text,
                                  per_timestep_conditioning_delta=per_timestep_conditioning_delta,
                                  per_timestep_uncond_delta=per_timestep_uncond_delta,
                                  xa_interpolation_schedule=xa_interpolation_schedule,
                                  self_interpolation_schedule=self_interpolation_schedule, solver_order=solver_order)

        #freeze non optim parameters
        for module in [self.unet, self.vae, self.text_encoder]:
            for param in module.parameters():
                param.requires_grad_(False)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if starting_img is not None:
            starting_latent = self.encode_image_to_latent(starting_img)
        else:
            starting_latent = None

        regularizers_fs_ws_names, collect_xa, requires_xa_foreground_mask\
            = self.setup_regularizers(regularizers_weights, starting_img, starting_latent)

        imgs_cpu = torch.zeros((sgd_steps + 1, 3, height, width))
        loss_scores = []

        regularizer_scores = {}
        for (_, _, reg_name) in regularizers_fs_ws_names:
            regularizer_scores[reg_name] = []

        # 7. SGD loop
        xa_store_initial_attention = requires_xa_foreground_mask or xa_interpolation_schedule is not None
        xa_store_last_attention = collect_xa
        self_store_initial_attention = self_interpolation_schedule is not None
        self.unet = prepare_unet(self.unet, xa_store_initial_attention_map=xa_store_initial_attention,
                                 xa_store_last_attention_map=xa_store_last_attention,
                                 self_store_initial_attention_map=self_store_initial_attention,
                                 self_store_last_attention_map=False, store_in_ram=True, store_dtype=None)

        with torch.no_grad():
            latents_xa_foreground_mask, px_xa_foreground_mask, reference_xa_maps, words_attention_masks, initial_image \
                = self.initial_denoising_loop(ddim_loop, initial_latents, conditional_embeds, conditional_embeds_delta,
                                              uncond_embeds, uncond_embeds_delta, collect_xa, foreground_token_mask,
                                              requires_xa_foreground_mask, timesteps, width,
                                              device, word_to_token_embeddings, squash_function, extra_step_kwargs)

            if prompt_to_prompt_replacements is not None:
                p2p_reshape_initial_cross_attention(self.unet, timesteps, word_to_token_embeddings,
                                                    p2p_word_to_token_embeddings, conditional_embeds,
                                                    p2p_conditional_embeds, prompt_to_prompt_replacements, device)
                conditional_embeds = p2p_conditional_embeds

            for outer_iteration in range(sgd_steps + 1):
                #calculate gradient of loss wrt to last latent x0
                with torch.enable_grad():
                    intermediate_preds, xa_maps = ddim_loop(uncond_embeds, conditional_embeds,
                                                 uncond_embeds_delta, conditional_embeds_delta,
                                                 initial_latents, extra_step_kwargs, collect_xa=collect_xa)


                    with torch.no_grad():
                        non_augmented_loss, image = self.calculate_loss(intermediate_preds, conditional_embeds_delta,
                                                                        uncond_embeds_delta, loss_w, loss_function,
                                                                        targets, xa_maps, reference_xa_maps,
                                                                        latents_xa_foreground_mask,
                                                                        px_xa_foreground_mask, regularizers_fs_ws_names,
                                                                        loss_steps, loss_steps_schedule,
                                                                        loss_scores=loss_scores,
                                                                        regularizer_scores=regularizer_scores,
                                                                        augment=False)
                    imgs_cpu[outer_iteration] = image.detach().cpu()
                    if outer_iteration == sgd_steps:
                        break

                    loss, _ = self.calculate_loss(intermediate_preds, conditional_embeds_delta, uncond_embeds_delta,
                                                  loss_w, loss_function, targets, xa_maps, reference_xa_maps,
                                                  latents_xa_foreground_mask, px_xa_foreground_mask,
                                                  regularizers_fs_ws_names, loss_steps, loss_steps_schedule,
                                                  augment=True)

                print_string = f'{outer_iteration} - Loss w. reg: {loss.item():.5f}' \
                               f' - Loss w. reg non augmented: {non_augmented_loss.item():.5f}' \
                               f' - Loss wo. reg: {loss_scores[-1]:.5f} '
                for reg_name, reg_s in regularizer_scores.items():
                    print_string += f' - {reg_name}: {reg_s[-1]:.5f}'

                if early_stopping_loss is not None and loss < early_stopping_loss:
                    imgs_cpu = imgs_cpu[:(1 + outer_iteration)]
                    print('Early stopping criterion reached')
                    break

                print(print_string)
                loss.backward()

                if gradient_clipping is not None:
                    for optim_variable_dict in optim.param_groups:
                        for var in optim_variable_dict['params']:
                            torch.nn.utils.clip_grad_norm_(var, gradient_clipping, 'inf')
                if normalize_gradients:
                    for optim_variable_dict in optim.param_groups:
                        for var in optim_variable_dict['params']:
                            grad_non_normalized = var.grad
                            grad_norm = torch.norm(grad_non_normalized.view(-1), p=2).item()
                            var.grad /= grad_norm

                #update parameters
                optim.step()
                if optimize_latents:
                    scale_factor = initial_latents_norm / torch.norm(initial_latents.view(-1), p=2)
                    initial_latents.mul_(scale_factor)
                if optim_scheduler is not None:
                    optim_scheduler.step()
                optim.zero_grad()
        free_unet(self.unet)

        return_values = {
            'imgs': imgs_cpu,
            'loss_scores': loss_scores,
            'regularizer_scores': regularizer_scores,
            'px_foreground_segmentation': px_xa_foreground_mask.cpu() if px_xa_foreground_mask is not None else None,
            'latents_foreground_segmentation': latents_xa_foreground_mask.cpu() if latents_xa_foreground_mask is not None else None,
            'words_attention_masks': words_attention_masks,
            'initial_img': initial_image,
        }

        return return_values

    def initial_denoising_loop(self, ddim_loop, initial_latents, conditional_embeds, conditional_embeds_delta,
                               uncond_embeds, uncond_embeds_delta, collect_xa, foreground_token_mask,
                               requires_xa_foreground_mask, timesteps, width, device, word_to_token_embeddings,
                               squash_function, extra_step_kwargs):

        latents_xa_foreground_mask = None
        px_xa_foreground_mask = None
        reference_xa_maps = None
        words_attention_masks = None
        initial_img = None
        with torch.no_grad():
            if collect_xa or requires_xa_foreground_mask:
                intermediate_latents, _ = ddim_loop(uncond_embeds, conditional_embeds,
                                             uncond_embeds_delta, conditional_embeds_delta,
                                             initial_latents, extra_step_kwargs, collect_xa=collect_xa)

                # save first cross attention map, used for segmentation calculation or prompt-2-prompt style xa editing
                if collect_xa or requires_xa_foreground_mask:
                    reference_xa_maps = get_initial_cross_attention_all_timesteps(self.unet, timesteps)
                # calculate latent "segmentation" based on foreground word
                if requires_xa_foreground_mask:
                    latents_xa_foreground_mask, px_xa_foreground_mask, words_attention_masks = \
                        self.calculate_xa_foreground_segmentation(foreground_token_mask, width // self.vae_scale_factor,
                                                                  width, reference_xa_maps, word_to_token_embeddings,
                                                                  squash_function=squash_function)
                    latents_xa_foreground_mask = latents_xa_foreground_mask.to(device)
                    px_xa_foreground_mask = px_xa_foreground_mask.to(device)

                latents = next(reversed(intermediate_latents.values()))
                initial_img = self.decode_latent_to_img(latents).detach().cpu().squeeze(dim=0)

        return latents_xa_foreground_mask, px_xa_foreground_mask, reference_xa_maps, words_attention_masks, initial_img

    def calculate_xa_foreground_segmentation(self, foreground_token_mask, latent_dimension, px_dimension, xa_maps,
                                             word_to_token_embeddings, xa_map_resolution=16, start_t=0.75, end_t=1.0,
                                             squash_function=None, ):
        matching_dim_xa_maps = []
        xa_map_num_pixels = xa_map_resolution ** 2
        for t_idx, ca_maps_t in enumerate(xa_maps.values()):
            t_current = 1 - t_idx / (len(xa_maps.keys()) - 1)
            if t_current < start_t or t_current > end_t:
                continue

            for ca_map_t in ca_maps_t.values():
                h = ca_map_t.shape[0]
                if ca_map_t.shape[1] == xa_map_num_pixels:
                    #only use conditional XA part
                    if XA_STORE_INITIAL_CONDITIONAL_ONLY:
                        matching_dim_xa_maps.append(ca_map_t)
                    else:
                        matching_dim_xa_maps.append(ca_map_t[h // 2:])
        stacked_xa_maps = torch.cat(matching_dim_xa_maps, dim=0)

        latent_foreground_segmentation = None
        px_foreground_segmentation = None
        words_attention_masks = {}
        for i in range(len(word_to_token_embeddings) + 1):
            if i == 0:
                stacked_xa_maps_foreground = stacked_xa_maps[:, :, foreground_token_mask.to(stacked_xa_maps.device)]
            else:
                word = list(word_to_token_embeddings.keys())[i - 1]
                token_embeddings_word = word_to_token_embeddings[word]
                word_token_mask = torch.zeros_like(foreground_token_mask)
                word_token_mask[torch.LongTensor(token_embeddings_word)] = 1
                stacked_xa_maps_foreground = stacked_xa_maps[:, :, word_token_mask.to(stacked_xa_maps.device)]

            xa_map_segmentation = stacked_xa_maps_foreground.mean(dim=2).mean(dim=0)
            xa_map_segmentation /= (xa_map_segmentation.max() + 1e-8)
            xa_map_segmentation_spatial = xa_map_segmentation.view(1, xa_map_resolution, xa_map_resolution)


            latent_xa_map_segmentation_spatial = torch.clamp(
                TF.resize(xa_map_segmentation_spatial, size=[latent_dimension, latent_dimension],
                          interpolation=TF.InterpolationMode.BICUBIC), 0.0, 1.0)
            px_xa_map_segmentation_spatial = torch.clamp(
                TF.resize(xa_map_segmentation_spatial, size=[px_dimension, px_dimension],
                          interpolation=TF.InterpolationMode.BICUBIC), 0.0, 1.0)
            if i == 0:

                latent_foreground_segmentation = apply_segmentation_postprocessing(squash_function, latent_xa_map_segmentation_spatial)
                px_foreground_segmentation = apply_segmentation_postprocessing(squash_function, px_xa_map_segmentation_spatial)
            else:
                words_attention_masks[word] = px_xa_map_segmentation_spatial.detach().cpu()
        return latent_foreground_segmentation, px_foreground_segmentation, words_attention_masks

    def create_optimizer(self, initial_latents, uncond_embeds_delta, conditional_embeds_delta, optimize_latents,
                         optimize_uncond, optimize_conditioning, sgd_optim, stepsize, latent_lr_factor=1.0,
                         conditioning_lr_factor=1.0, uncond_lr_factor=1.0):
        optim_variables = []
        assert optimize_conditioning or optimize_latents or optimize_uncond
        if optimize_latents:
            initial_latents.requires_grad_(True)
            optim_variables.append({'params': [initial_latents], "lr": stepsize * latent_lr_factor})
        if optimize_conditioning:
            conditional_embeds_delta.requires_grad_(True)
            optim_variables.append({'params': [conditional_embeds_delta], "lr": stepsize * conditioning_lr_factor})
        if optimize_uncond:
            uncond_embeds_delta.requires_grad_(True)
            optim_variables.append({'params': [uncond_embeds_delta], "lr": stepsize * uncond_lr_factor})
        if sgd_optim == 'adam':
            optim = torch.optim.Adam(optim_variables, lr=stepsize)
        elif sgd_optim == 'adamw':
            optim = torch.optim.AdamW(optim_variables, lr=stepsize)
        elif sgd_optim == 'sgd':
            optim = torch.optim.SGD(optim_variables, lr=stepsize, momentum=0.9)
        else:
            raise NotImplementedError()

        return optim

    def setup_regularizers(self, regularizers_weights, starting_img, starting_latent):
        regularizers_fs_ws_names = []
        collect_xa = False
        requires_xa_foreground_mask = False
        for regularizer_name, regularizer_w in regularizers_weights.items():
            if regularizer_name == 'px_l1':
                def reg(**kwargs):
                    image = kwargs['image'].squeeze()
                    px_dist = F.l1_loss(image, starting_img, reduction='mean')
                    return px_dist
            elif regularizer_name == 'px_l2':
                def reg(**kwargs):
                    image = kwargs['image'].squeeze()
                    px_dist = F.mse_loss(image, starting_img, reduction='mean')
                    return px_dist
            elif regularizer_name == 'px_lpips':
                loss_fn_alex = LPIPS(net='alex').to(starting_img.device)
                def reg(**kwargs):
                    image = kwargs['image']
                    px_dist = loss_fn_alex(image, starting_img[None, :, :, :], normalize=True).mean()
                    return px_dist
            elif regularizer_name == 'latent_l1':
                def reg(**kwargs):
                    latents = kwargs['latents']
                    latent_dist = F.l1_loss(latents.view(-1), starting_latent.view(-1), reduction='mean')
                    return latent_dist
            elif regularizer_name == 'latent_l2':
                def reg(**kwargs):
                    latents = kwargs['latents']
                    latent_dist = F.mse_loss(latents.view(-1), starting_latent.view(-1), reduction='mean')
                    return latent_dist
            elif regularizer_name in ['px_foreground_l2', 'px_background_l2', 'px_foreground_l1', 'px_background_l1']:
                requires_xa_foreground_mask = True

                def reg_fn(regularizer_name, **kwargs):
                    xa_foreground_mask = kwargs['px_xa_foreground_mask']
                    if 'background' in regularizer_name:
                        mask = (1. - xa_foreground_mask)
                    elif 'foreground' in regularizer_name:
                        mask = xa_foreground_mask
                    else:
                        raise NotImplementedError()

                    image = kwargs['image'].squeeze()
                    if 'l2' in regularizer_name:
                        px_dist = torch.mean(mask * (image - starting_img)**2)
                    elif 'l1' in regularizer_name:
                        px_dist = torch.mean(mask * (image - starting_img).abs())
                    else:
                        raise NotImplementedError()

                    return px_dist

                reg = partial(reg_fn, regularizer_name)
            elif regularizer_name in ['px_foreground_lpips', 'px_background_lpips']:
                requires_xa_foreground_mask = True
                loss_fn_alex_masked = MaskedLPIPS(net='alex').to(starting_img.device)
                def reg_fn(regularizer_name, **kwargs):
                    xa_foreground_mask = kwargs['px_xa_foreground_mask']
                    if 'background' in regularizer_name:
                        mask = (1. - xa_foreground_mask)
                    elif 'foreground' in regularizer_name:
                        mask = xa_foreground_mask
                    else:
                        raise NotImplementedError()

                    image = kwargs['image']
                    px_dist = loss_fn_alex_masked(image, starting_img[None, :, :, :],
                                                  mask[None, :, :, :], normalize=True).mean()
                    return px_dist

                reg = partial(reg_fn, regularizer_name)
            elif regularizer_name in ['latent_foreground_l2', 'latent_background_l2', 'latent_foreground_l1', 'latent_background_l1']:
                requires_xa_foreground_mask = True
                def reg_fn(regularizer_name, **kwargs):
                    latents = kwargs['latents']
                    xa_foreground_mask = kwargs['latents_xa_foreground_mask']
                    if 'background' in regularizer_name:
                        mask = (1. - xa_foreground_mask[None, :])
                    elif 'foreground' in regularizer_name:
                        mask = xa_foreground_mask[None, :]
                    else:
                        raise NotImplementedError()

                    if 'l2' in regularizer_name:
                        latent_dist = torch.mean(mask * (latents - starting_latent)**2)
                    elif 'l1' in regularizer_name:
                        latent_dist = torch.mean(mask * (latents - starting_latent).abs())
                    else:
                        raise NotImplementedError()

                    return latent_dist

                reg = partial(reg_fn, regularizer_name)
            else:
                raise NotImplementedError(regularizer_name)

            regularizers_fs_ws_names.append((reg, regularizer_w, regularizer_name))

        return regularizers_fs_ws_names, collect_xa, requires_xa_foreground_mask

    def calculate_loss(self, intermediate_preds, conditional_embeds_delta, uncond_embeds_delta, loss_w, loss_function,
                       targets, xa_maps, reference_xa_maps, latents_xa_foreground_mask, px_xa_foreground_mask,
                       regularizers_fs_ws_names, loss_steps, loss_steps_schedule, loss_scores=None,
                       regularizer_scores=None, augment=True):

        loss = 0

        for i, latents in intermediate_preds.items():
            if not augment:
                #without augment we only evaluate the loss value of the final image
                if i < (len(intermediate_preds) - 1):
                    continue
                else:
                    loss_w_i = 1.0
            else:
                if i < (len(intermediate_preds) - loss_steps):
                    continue
                else:
                    if loss_steps_schedule == 'uniform':
                        loss_w_i = 1 / loss_steps
                    elif loss_steps_schedule == 'linear':
                        loss_w_i = (1 + loss_steps - len(intermediate_preds) + i) / sum(range(1 + loss_steps))
                    else:
                        raise ValueError()

            latents = latents.to(torch.float32)
            image = self.decode_latent_to_img(latents)

            if loss_function is not None:
                main_loss = loss_function(image, targets, augment=augment)
                loss += loss_w * loss_w_i * main_loss
                if loss_scores is not None and (i == len(intermediate_preds) - 1):
                    loss_scores.append(main_loss.detach().item())

            regularizer_kwargs = {
                'image': image,
                'latents': latents,
                'conditional_embeds_delta': conditional_embeds_delta,
                'uncond_embeds_delta': uncond_embeds_delta,
                'reference_xa': reference_xa_maps,
                'xa': xa_maps,
                'latents_xa_foreground_mask': latents_xa_foreground_mask,
                'px_xa_foreground_mask': px_xa_foreground_mask,
            }

            for (regularizer_f, regularizer_w, regularizer_name) in regularizers_fs_ws_names:
                reg_score = regularizer_f(**regularizer_kwargs)
                loss = loss + loss_w_i * regularizer_w * reg_score
                if regularizer_scores is not None and (i == len(intermediate_preds) - 1):
                    regularizer_scores[regularizer_name].append(reg_score.detach().item())

        return loss, image

    #modified encoding that returns the positions of words in encoding that each word in input corresponds to
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None:
            raise NotImplementedError()

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            #positions
            enc = {x: self.tokenizer(x,
                padding=False,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )['input_ids'] for x in prompt.split()}

            word_to_token_embeddings = {}
            #clip encoding always has a 49406/49407 at the start/end of sentence which does not correspond to an input word
            token_map_idx = 1
            for word, token in enc.items():
                tokenoutput = []
                for ids in token.view(-1):
                    #ignore start/end of sentence tokens
                    if ids.item() != 49406 and ids.item() != 49407:
                        tokenoutput.append(token_map_idx)
                        token_map_idx += 1
                word_to_token_embeddings[word] = tokenoutput

            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds, word_to_token_embeddings




