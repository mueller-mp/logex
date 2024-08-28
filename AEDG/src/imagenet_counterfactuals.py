import torch

from diffusers import DDIMScheduler

import os
from omegaconf import OmegaConf
import wandb
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime


from typing import Dict, List, Optional
from dataclasses import dataclass, field
from utils.models.load_timm_model import load_timm_model, load_timm_model_with_cutout, load_timm_model_with_random_noise
from utils.sd_backprop_pipe import StableDiffusionPipelineWithGrad
from utils.plotting_utils import plot, plot_attention
from utils.loss_functions import get_loss_function, calculate_confs, calculate_lp_distances, calculate_lpips_distances
from utils.inet_classes import IDX2NAME as in_labels
from utils.inet_hierarchy import find_cluster_for_class_idx
from utils.parser_utils import CommonArguments
from utils.inet_vce_idcs import VCE_start_class_target
from utils.wandb_utils import make_wandb_run

def fill_img_idcs(img_id_start_target, inversion_folder, num_ddim_steps, guidance_scale):
    new_id_start_target = []
    for img_idx, start_class, target_class in img_id_start_target:
        #check if inversion exists
        start_label = in_labels[start_class]
        start_class_folder = os.path.join(inversion_folder, f'{start_class}_{start_label}')
        inversion_dir = os.path.join(start_class_folder,  f"inversion_{num_ddim_steps}_{guidance_scale}")

        if not os.path.isfile(os.path.join(inversion_dir, f'{img_idx}.pt')):
            print(f'Could not find inversion file {img_idx} {start_class}')
            continue

        if target_class is not None:
            new_id_start_target.append((img_idx, start_class, target_class))
        else:
            #find matching in cluster
            in_cluster = find_cluster_for_class_idx(start_class)
            if in_cluster is None:
                print(f'Not cluster found for class {start_class}; Skipping')
            else:
                for t_idx, _ in in_cluster:
                    if t_idx != start_class:
                        new_id_start_target.append((img_idx, start_class, t_idx))

    return new_id_start_target


@dataclass
class CounterfactualArgs(CommonArguments):
    gpu: int = 0
    seed: int = 42
    num_images: Optional[int] = None
    local_split: int = 0
    num_splits: int = 0

    classifier: str = 'vit_base_patch16_384.augreg_in21k_ft_in1k'

    results_folder: str = 'output_cvpr/imagenet_counterfactuals'
    results_sub_folder: Optional[str] = None
    inversion_folder: str = 'output_cvpr/imagenet_inversions'
    imagenet_folder: str = '/mnt/datasets/imagenet'

    regularizers: List[str] = field(default_factory=lambda: [])
    regularizers_ws: List[float] = field(default_factory=lambda: [])
    segmentation_squash: Optional[str] = None

    loss: str = 'log_conf'

    p2p: bool = True
    xa_interpolation_schedule: Optional[str] = 'cosine_1.5'
    self_interpolation_schedule: Optional[str] = None
    xa_interp_softmax: bool = False


def setup() -> CounterfactualArgs:
    default_config: CounterfactualArgs = OmegaConf.structured(CounterfactualArgs)
    cli_args = OmegaConf.from_cli()
    config: CounterfactualArgs = OmegaConf.merge(default_config, cli_args)
    return config


if __name__=="__main__":
    args = setup()

    #torch.set_deterministic_debug_mode(1)
    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(args.seed)

    classifier_name = args.classifier
    #classifier_name = 'vit_large_patch16_224'
    #classifier_name = 'tv_resnet50'
    num_cutouts = args.augmentation_num_cutouts
    if num_cutouts > 0:
        augm_string = f'_cutout_{num_cutouts}'
        classifier = load_timm_model_with_cutout(classifier_name, 0.3, num_cutouts, checkpointing=True,
                                                 noise_sd=args.augmentation_noise_sd,
                                                 noise_schedule=args.augmentation_noise_schedule,
                                                 noise_descending_steps=args.sgd_steps)
    else:
        augm_string = '_no_aug'
        classifier = load_timm_model(classifier_name)

    if args.use_double:
        classifier.double()
    classifier.to(device)

    pipe = StableDiffusionPipelineWithGrad.from_pretrained(args.model_path)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if args.use_double:
        pipe.convert_to_double()

    pipe.to(device)

    loss_function = get_loss_function('log_conf', classifier)
    to_tensor = transforms.ToTensor()

    assert len(args.regularizers) == len(args.regularizers_ws)
    regularizers_weights = {}
    reg_string = ''
    for reg_name, reg_w in zip(args.regularizers, args.regularizers_ws):
        regularizers_weights[reg_name] = reg_w
        reg_string += f'_{reg_name}_{reg_w}'

    to_pil = transforms.ToPILImage()

    inversion_root_dir = os.path.join(args.inversion_folder, args.model_path)
    img_id_start_target = fill_img_idcs(VCE_start_class_target, inversion_root_dir,
                                        args.num_ddim_steps, args.guidance_scale)
    if args.num_images:
        selected_subset = torch.randperm(len(img_id_start_target))[:args.num_images]
        img_id_start_target = [img_id_start_target[i] for i in selected_subset]

    # datetime object containing current date and time
    if args.results_sub_folder is None:
        output_description = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    else:
        output_description = args.results_sub_folder

    result_folder = os.path.join(args.results_folder, output_description)
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, 'config.yaml'), "w") as f:
        OmegaConf.save(args, f)

    make_wandb_run(args.wandb_project, output_description, OmegaConf.to_container(args))

    for linear_idx, (img_idx, start_class, target_class) in enumerate(img_id_start_target):
        if args.num_splits > 1:
            current_split_idx = linear_idx % args.num_splits
            if not current_split_idx == args.local_split:
                continue

        start_label = in_labels[start_class]
        target_label = in_labels[target_class]

        if args.p2p:
            prompt_to_prompt_replacements = (start_label, target_label)
        else:
            prompt_to_prompt_replacements = None
        start_class_folder = os.path.join(inversion_root_dir, f'{start_class}_{start_label}')
        inversion_dir = os.path.join(start_class_folder,
                                     f"inversion_{args.num_ddim_steps}_{args.guidance_scale}")

        result_folder = os.path.join(args.results_folder, output_description,
                                     f'{start_class}_{start_label}/')
        os.makedirs(result_folder, exist_ok=True)

        out_pdf = os.path.join(result_folder, f'{img_idx}_{target_class}_{target_label}.pdf')
        if os.path.isfile(out_pdf):
            continue

        latent = torch.load(os.path.join(inversion_dir, f'{img_idx}.pt'), map_location='cpu')[None, :]
        null_texts_embeddings = torch.load(os.path.join(inversion_dir, f'{img_idx}_null_texts.pt'), map_location='cpu')
        assert len(null_texts_embeddings) == args.num_ddim_steps

        if args.use_double:
            latent = latent.double()
            null_texts_embeddings = null_texts_embeddings.double()

        captions_file = os.path.join(inversion_dir,  f'{img_idx}_prompt.txt')
        if os.path.isfile(captions_file):
            with open(captions_file, 'r') as f:
                prompt_str = f.read()
        else:
            print(f'Warning: Could not load caption from {captions_file}')
            continue

        prompt_foreground_key_words = [word for word in start_label.split()]
        if True:
            prompt_to_prompt_str = prompt_str.replace(start_label, target_label)

        original_img = Image.open(os.path.join(inversion_dir, f'{img_idx}_original.png'))
        original_tensor = to_tensor(original_img).to(device)
        if args.use_double:
            original_tensor = original_tensor.double()

        reconstructed_img = Image.open(os.path.join(inversion_dir, f'{img_idx}_null_reconstructed.png'))
        reconstructed_tensor = to_tensor(reconstructed_img)

        return_values = pipe(target_class,
                             prompt=prompt_str,
                             prompt_foreground_key_words=prompt_foreground_key_words,
                             prompt_to_prompt_replacements=prompt_to_prompt_replacements,
                             squash_function=args.segmentation_squash,
                             starting_img=original_tensor,
                             latents=latent,
                             regularizers_weights=regularizers_weights,
                             sgd_steps=args.sgd_steps,
                             sgd_stepsize=args.sgd_stepsize,
                             latent_lr_factor=args.latent_lr_factor,
                             conditioning_lr_factor=args.conditioning_lr_factor,
                             uncond_lr_factor=args.uncond_lr_factor,
                             xa_interpolation_schedule=args.xa_interpolation_schedule,
                             self_interpolation_schedule=args.self_interpolation_schedule,
                             early_stopping_loss=None,
                             sgd_optim=args.optim,
                             sgd_scheduler=args.sgd_schedule,
                             normalize_gradients=args.normalize_gradient,
                             gradient_clipping=None,
                             loss_function=loss_function,
                             loss_steps=args.loss_steps,
                             loss_steps_schedule=args.loss_steps_schedule,
                             guidance_scale=args.guidance_scale,
                             optimize_latents=args.optimize_latents,
                             optimize_conditioning=args.optimize_conditioning,
                             optimize_uncond=args.optimize_uncond,
                             per_timestep_conditioning_delta=args.per_timestep_conditioning,
                             per_timestep_uncond_delta=args.per_timestep_uncond,
                             null_text_embeddings=null_texts_embeddings,
                             height=args.resolution, width=args.resolution,
                             num_inference_steps=args.num_ddim_steps)

        with torch.no_grad():
            img_grid = return_values['imgs']
            loss_scores = return_values['loss_scores']
            regularizer_scores = return_values['regularizer_scores']
            px_foreground_segmentation = return_values['px_foreground_segmentation']
            words_attention_masks = return_values['words_attention_masks']
            pre_p2p_image = return_values['initial_img']

            start_confs = calculate_confs(classifier, img_grid, device, target_class=start_class)
            target_confs = calculate_confs(classifier, img_grid, device, target_class=target_class)

            confs = torch.cat([start_confs[:, None], target_confs[:, None]], dim=1)

            total_losses = torch.FloatTensor(loss_scores)
            for reg_w, reg_score in zip(args.regularizers_ws, regularizer_scores.values()):
                total_losses += torch.FloatTensor(reg_score)

            min_loss_idx = torch.argmin(total_losses).item()

            min_loss_img = img_grid[min_loss_idx]
            #torch.save(min_loss_img, os.path.join(result_folder, f'{img_idx}.pth'))

            original_img.save(os.path.join(result_folder, f'{img_idx}.png'))
            pil_sd = to_pil(torch.clamp(img_grid[0], 0, 1))
            pil_sd.save(os.path.join(result_folder, f'{img_idx}_{target_class}_{target_label}_sd.png'))
            pil_ours = to_pil(torch.clamp(min_loss_img, 0, 1))
            pil_ours.save(os.path.join(result_folder, f'{img_idx}_{target_class}_{target_label}_ours.png'))


            title_attribute = ''
            max_class_name_length = 8
            if len(start_label) > max_class_name_length:
                title_attribute += start_label[:max_class_name_length] + '.'
            else:
                title_attribute += start_label
            title_attribute += ' - '
            if len(target_label) > max_class_name_length:
                title_attribute += target_label[:max_class_name_length] + '.'
            else:
                title_attribute += target_label

            plot(original_tensor, pre_p2p_image, img_grid, title_attribute, confs, out_pdf, loss_scores=total_losses,
                 regularizer_scores=regularizer_scores,
                 wandb_name=f'history' if args.wandb_project is not None else None,
                 wandb_step=linear_idx)

            plot_attention(original_tensor, px_foreground_segmentation, words_attention_masks,
                           os.path.join(result_folder, f'{img_idx}_attention.pdf'),
                           wandb_name=f'attention' if args.wandb_project is not None else None,
                           wandb_step=linear_idx)

            if args.wandb_project is not None:
                lp_distances = calculate_lp_distances([img_grid[0], min_loss_img], [original_tensor, original_tensor], ps=(1., 2.))
                lpips_distances = calculate_lpips_distances([img_grid[0], min_loss_img], [original_tensor, original_tensor])
                wandb_log_dict = {
                    'loss': loss_scores[min_loss_idx],
                    'l2_best': lp_distances[2.][1],
                    'l2_p2p': lp_distances[2.][0],
                    'l1_best': lp_distances[1.][1],
                    'l1_p2p': lp_distances[1.][0],
                    'lpips_best': lpips_distances[1],
                    'lpips_p2p': lpips_distances[0],
                    'start_class_conf': start_confs[min_loss_idx],
                    'target_class_conf': target_confs[min_loss_idx],
                    'start_p2p_ours': [wandb.Image(img, caption=cap) for img, cap in zip([original_img, pil_sd, pil_ours],
                                                                                         [prompt_str, prompt_to_prompt_str, ''])]
                }
                if regularizer_scores:
                    for reg_n, reg_score in regularizer_scores.items():
                        wandb_log_dict[reg_n] = reg_score
                wandb.log(wandb_log_dict,
                          step=linear_idx)

