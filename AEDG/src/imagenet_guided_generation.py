import math
import torch

from diffusers import DDIMScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler, DPMSolverMultistepScheduler, PNDMScheduler

import os
import argparse
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from utils.models.load_timm_model import load_timm_model, load_timm_model_with_cutout
from utils.models.load_shape_model import load_sin_model, load_sin_model_with_cutout
from utils.sd_backprop_pipe import StableDiffusionPipelineWithGrad
from utils.plotting_utils import plot
from utils.loss_functions import get_loss_function, calculate_confs
from utils.inet_classes import IDX2NAME
from utils.parser_utils import CommonArguments

def get_gradcam(classifier, classifier_name):
    try:
        from pytorch_grad_cam import EigenCAM, GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        if classifier_name == 'resnet50_sin':
            target_layers = [classifier.model.layer4[-1]]
        elif classifier_name == 'tv_resnet50':
            target_layers = [classifier.model.layer4[-1]]
        else:
            raise NotImplementedError(f'GradCAM target layer not specified for: {classifier_name}')

        cam = GradCAM(model=classifier, target_layers=target_layers, use_cuda=False)
        return cam, target_layers
    except:
        return None, None

#python src/imagenet_guided_generation.py --gpu 1 --guidance_scale 3.0 --sgd_steps 15 --num_images 100 --linesearch none --optimize_conditioning --optimize_uncond --optimize_latent --per_timestep_conditioning --per_timestep_uncond --latent_lr_factor 0.01 --mode all --target_classes 878 415 987 907 784 899 834 531 817 968 598 893 515 479 278 924 758 124 281 434 738 --sgd_stepsize 0.025 --sgd_schedule cosine --augmentation_num_cutouts 16 --augmentation_noise_sd 0.05 --augmentation_noise_schedule const --loss conf --num_ddim_steps 50 --solver ddim --classifier1 vit_large_patch16_384.augreg_in21k_ft_in1k --classifier2 convnext_base.fb_in1k --local_split 0 --num_splits 3
#python src/imagenet_guided_generation.py --gpu 6 --guidance_scale 3.0 --sgd_steps 15 --num_images 100 --linesearch none --optimize_conditioning --optimize_uncond --optimize_latent --per_timestep_conditioning --per_timestep_uncond --latent_lr_factor 0.01 --mode all --target_classes 58 309 313 400 415 416 419 424 433 434 435 440 465 514 557 585 626 737 746 752 --sgd_stepsize 0.025 --sgd_schedule cosine --augmentation_num_cutouts 16 --augmentation_noise_sd 0.05 --augmentation_noise_schedule const --loss conf --num_ddim_steps 50 --solver ddim --classifier2 vit_base_patch16_384.augreg_in21k_ft_in1k --classifier1 convnext_base.fb_in1k --local_split 0 --num_splits 3

#python src/imagenet_guided_generation.py gpu=6 guidance_scale=3.0 sgd_steps=15 num_images=100 optimize_conditioning=True optimize_uncond=True per_timestep_conditioning=True per_timestep_uncond=True latent_lr_factor=0.01 target_classes=[58,309] sgd_stepsize=0.025 sgd_schedule=cosine augmentation_num_cutouts=16 augmentation_noise_sd=0.05 augmentation_noise_schedule=const loss=conf num_ddim_steps=50 solver=ddim classifier1=resnet50 local_split=0 num_splits=3

@dataclass
class GuidedArgs(CommonArguments):
    gpu: int = 0
    seed: int = 42
    num_images: int = 10
    local_split: int = 0
    num_splits: int = 0

    classifier1: str = 'vit_large_patch16_224'
    classifier2: Optional[str] = None
    eval_classifier: Optional[str] = 'eva_large_patch14_336.in22k_ft_in22k_in1k'

    target_classes: Optional[List[int]] = None
    early_stopping_confidence: float = 1.0

    grad_cam: bool = False

    results_folder: str = 'output_cvpr/imagenet_guided'

    solver: str = 'ddim'

    regularizers: List[str] = field(default_factory=lambda: [])
    regularizers_ws: List[float] = field(default_factory=lambda: [])
    segmentation_squash: Optional[str] = None

    loss: str = 'log_conf'

def setup() -> GuidedArgs:
    default_config: GuidedArgs = OmegaConf.structured(GuidedArgs)
    cli_args = OmegaConf.from_cli()
    config: GuidedArgs = OmegaConf.merge(default_config, cli_args)
    return config

if __name__=="__main__":
    args = setup()
    device = torch.device(f'cuda:{args.gpu}')

    if args.target_classes:
        class_sort_idcs = torch.LongTensor(args.target_classes)
    else:
        class_sort_idcs = torch.arange(0, 1000, dtype=torch.long)

    loss = args.loss

    pipe = StableDiffusionPipelineWithGrad.from_pretrained(args.model_path)
    if args.solver == 'ddim':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        solver_order = 1
    elif args.solver == 'heun':
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
        solver_order = 2
    elif args.solver == 'dpm':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        solver_order = 2
    elif args.solver == 'pndm':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        solver_order = 2
    else:
        raise NotImplementedError()

    if args.use_double:
        pipe.convert_to_double()

    pipe.to(device)

    classifier_name = args.classifier1
    num_cutouts = args.augmentation_num_cutouts
    if classifier_name == 'resnet50_sin':
        if num_cutouts > 0:
            augm_string = f'_cutout_{num_cutouts}'
            classifier = load_sin_model_with_cutout(0.3, num_cutouts, checkpointing=True,
                                                    noise_sd=args.augmentation_noise_sd,
                                                    noise_schedule=args.augmentation_noise_schedule,
                                                    noise_descending_steps=args.sgd_steps)
        else:
            augm_string = '_no_aug'
            classifier = load_sin_model()
    else:
        if num_cutouts > 0:
            augm_string = f'_cutout_{num_cutouts}'
            classifier = load_timm_model_with_cutout(classifier_name, 0.3, num_cutouts, checkpointing=True,
                                                     noise_sd=args.augmentation_noise_sd,
                                                     noise_schedule=args.augmentation_noise_schedule,
                                                     noise_descending_steps=args.sgd_steps)
        else:
            augm_string = '_no_aug'
            classifier = load_timm_model(classifier_name)

    if args.grad_cam:
        classifier_cam, classifier_cam_target_layers = get_gradcam(classifier, classifier_name)
    else:
        classifier_cam = None
        classifier_cam_target_layers = None

    if args.augmentation_noise_sd > 0:
        augm_string += f'_noise_{args.augmentation_noise_schedule}_{args.augmentation_noise_sd}'

    classifier.to(device)
    eval_classifier_name = args.eval_classifier
    eval_classifier = load_timm_model(eval_classifier_name)
    eval_classifier.to(device)

    if args.classifier2 is not None:
        loss = 'conf'
        classifier2_name = args.classifier2
        if classifier2_name == 'resnet50_sin':
            if num_cutouts > 0:
                classifier2 = load_sin_model_with_cutout(0.3, num_cutouts, checkpointing=True, noise_sd=args.augmentation_noise_sd,
                                                    noise_schedule=args.augmentation_noise_schedule,
                                                    noise_descending_steps=args.sgd_steps)
            else:
                classifier2 = load_sin_model()
        else:
            if num_cutouts > 0:
                classifier2 = load_timm_model_with_cutout(classifier2_name, 0.3, num_cutouts, checkpointing=True, noise_sd=args.augmentation_noise_sd,
                                                    noise_schedule=args.augmentation_noise_schedule,
                                                    noise_descending_steps=args.sgd_steps)
            else:
                classifier2 = load_timm_model(classifier2_name)
        classifier2 = classifier2.to(device)

        result_folder_pre = f'{args.results_folder}_difference_{classifier_name}_{classifier2_name}'
        early_stopping_loss = None#-args.early_stopping_confidence

        if args.grad_cam:
            classifier2_cam, classifier2_cam_target_layers = get_gradcam(classifier2, classifier2_name)
        else:
            classifier2_cam = None
            classifier2_cam_target_layers = None
    else:
        classifier2 = None
        classifier2_name = None
        classifier2_cam = None
        classifier2_cam_target_layers = None
        result_folder_pre = f'{args.results_folder}_all'
        early_stopping_loss = -math.log(args.early_stopping_confidence)

    print(f'Writing results to: {result_folder_pre}')

    loss_function = get_loss_function(loss, classifier, classifier2=classifier2)
    to_tensor = transforms.ToTensor()

    guidance_scale = args.guidance_scale
    sgd_steps = args.sgd_steps
    sgd_stepsize = args.sgd_stepsize
    sgd_schedule = args.sgd_schedule
    optimize_latents = args.optimize_latents
    optimize_conditioning = args.optimize_conditioning
    optimize_uncond = args.optimize_uncond
    per_timestep_conditioning_delta = args.per_timestep_conditioning
    per_timestep_uncond_delta = args.per_timestep_uncond
    latent_lr_factor = args.latent_lr_factor
    conditioning_lr_factor = 1.0 if per_timestep_conditioning_delta else 1.0
    uncond_lr_factor = 1.0 if per_timestep_uncond_delta else 1.0

    latent_dim = (pipe.unet.config.in_channels, args.resolution // pipe.vae_scale_factor, args.resolution // pipe.vae_scale_factor)
    torch.manual_seed(0)
    num_classes = 1000
    deterministic_latents = torch.randn((num_classes, args.num_images if args.num_images < 50 else 50) + latent_dim, dtype=torch.float)

    output_description = f'{args.solver}_{guidance_scale}_{args.num_ddim_steps}_{loss}_sgd_{sgd_steps}_{sgd_stepsize}_{sgd_schedule}'
    if args.normalize_gradient:
        output_description += '_normalize'
    if optimize_latents:
        output_description += f'_latents'
        output_description += f'_{latent_lr_factor}' if latent_lr_factor != 1.0 else ''
    if optimize_conditioning:
        output_description += f'_cond'
        output_description += '_pt' if per_timestep_conditioning_delta else ''
        output_description += f'_{conditioning_lr_factor}' if conditioning_lr_factor != 1.0 else ''
    if optimize_uncond:
        output_description += f'_uncond'
        output_description += '_pt' if per_timestep_uncond_delta else ''
        output_description += f'_{uncond_lr_factor}' if uncond_lr_factor != 1.0 else ''
    output_description += augm_string

    to_pil = transforms.ToPILImage()

    result_folder = os.path.join(result_folder_pre, output_description)
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, 'config.yaml'), "w") as f:
        OmegaConf.save(args, f)

    for img_idx in range(args.num_images):
        for class_i, class_idx in enumerate(class_sort_idcs):
            class_idx = class_idx.item()
            if args.num_splits > 1:
                current_split_idx = class_i % args.num_splits
                if not current_split_idx == args.local_split:
                    continue

            if class_idx == 0:
                continue

            target_label = IDX2NAME[class_idx]
            target_class = class_idx

            result_folder_postfix = f'{class_idx}_{class_idx}_{target_label}'
            result_folder = os.path.join(result_folder_pre, output_description, result_folder_postfix)
            out_file_pth = os.path.join(result_folder, f'{img_idx}.pdf')
            if os.path.isfile(out_file_pth):
                continue
            os.makedirs(result_folder, exist_ok=True)

            target_preds = torch.zeros(args.num_images, dtype=torch.long)
            target_eval_preds = torch.zeros(args.num_images, dtype=torch.long)

            latent = deterministic_latents[class_idx, img_idx][None,:]
            null_texts_embeddings = None
            target_prompt_str = None
            original_tensor = None
            reconstructed_tensor = None

            prompt_str = f'a photograph of a {target_label}'

            return_values = pipe(target_class,
                                 prompt=prompt_str,
                                 starting_img=original_tensor,
                                 latents=latent,
                                 sgd_steps=sgd_steps,
                                 sgd_stepsize=sgd_stepsize,
                                 latent_lr_factor=latent_lr_factor,
                                 conditioning_lr_factor=conditioning_lr_factor,
                                 loss_steps=args.loss_steps,
                                 loss_steps_schedule=args.loss_steps_schedule,
                                 uncond_lr_factor=uncond_lr_factor,
                                 early_stopping_loss=early_stopping_loss,
                                 sgd_optim=args.optim,
                                 sgd_scheduler=sgd_schedule,
                                 normalize_gradients=args.normalize_gradient,
                                 gradient_clipping=None,
                                 solver_order=solver_order,
                                 loss_function=loss_function,
                                 guidance_scale=guidance_scale,
                                 optimize_latents=optimize_latents,
                                 optimize_conditioning=optimize_conditioning,
                                 optimize_uncond=optimize_uncond,
                                 per_timestep_conditioning_delta=per_timestep_conditioning_delta,
                                 per_timestep_uncond_delta=per_timestep_uncond_delta,
                                 null_text_embeddings=null_texts_embeddings,
                                 height=args.resolution, width=args.resolution,
                                 num_inference_steps=args.num_ddim_steps)

            img_grid = return_values['imgs']
            print(img_grid.shape)
            loss_scores = return_values['loss_scores']
            regularizer_scores = return_values['regularizer_scores']

            confs, preds = calculate_confs(classifier, img_grid, device, target_class=target_class, return_predictions=True)
            eval_confs, eval_preds = calculate_confs(eval_classifier, img_grid, device, target_class=target_class, return_predictions=True)

            if classifier2 is not None:
                confs2, preds2 = calculate_confs(classifier2, img_grid, device, target_class=target_class,
                                               return_predictions=True)
                merged_confs = torch.cat([confs[:, None], confs2[:, None], eval_confs[:, None]], dim=1)
            else:
                merged_confs = torch.cat([confs[:, None], eval_confs[:, None]], dim=1)

            classifier_imgs_cams_maps = None
            if classifier_cam is not None:
                num_classifiers = 1 if classifier2 is None else 2


                for layer in classifier_cam_target_layers:
                    for param in layer.parameters():
                        param.requires_grad_(True)

                classifier_imgs_cams_maps = torch.zeros((num_classifiers, len(img_grid), 1, args.resolution, args.resolution))
                targets = [ClassifierOutputTarget(target_class)]
                for cam_i, img in enumerate(img_grid):
                    cam_map = classifier_cam(input_tensor=img[None,:].to(device).requires_grad_(True), targets=targets)
                    classifier_imgs_cams_maps[0, cam_i] = torch.from_numpy(cam_map)[None, :, :]

                for layer in classifier_cam_target_layers:
                    for param in layer.parameters():
                        param.requires_grad_(False)

                if classifier2 is not None:
                    for layer in classifier2_cam_target_layers:
                        for param in layer.parameters():
                            param.requires_grad_(True)

                    for cam_i, img in enumerate(img_grid):
                        cam_map = classifier2_cam(input_tensor=img[None, :].to(device).requires_grad_(True),
                                                 targets=targets)
                        classifier_imgs_cams_maps[1, cam_i] = torch.from_numpy(cam_map)[None, :, :]

                    for layer in classifier2_cam_target_layers:
                        for param in layer.parameters():
                            param.requires_grad_(False)

            min_loss_idx = torch.argmin(torch.FloatTensor(loss_scores)).item()
            target_preds[img_idx] = preds[min_loss_idx]
            target_eval_preds[img_idx] = eval_preds[min_loss_idx]

            min_loss_img = img_grid[min_loss_idx]
            pil_sd = to_pil(torch.clamp(img_grid[0], 0, 1))
            pil_sd = pil_sd.save(os.path.join(result_folder, f'{img_idx}_sd.png'))
            pil_ours = to_pil(torch.clamp(min_loss_img, 0, 1))
            pil_ours = pil_ours.save(os.path.join(result_folder, f'{img_idx}_ours.png'))

            plot(original_tensor, reconstructed_tensor, img_grid, target_label, merged_confs,
                 os.path.join(result_folder, f'{img_idx}.pdf'), loss_scores=loss_scores,
                 regularizer_scores=regularizer_scores, model_grad_cams=classifier_imgs_cams_maps)

