import math
import torch

from diffusers import DDIMScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler, DPMSolverMultistepScheduler, PNDMScheduler

import os
import argparse
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from torchvision.transforms import functional as TF

from utils.models.load_robust_model import load_madry_l2_with_cutout
from utils.models.load_vits import load_vit_with_cutout, load_vit
from utils.models.load_tv import load_tv_with_cutout, load_tv
from utils.sd_backprop_pipe import StableDiffusionPipelineWithGrad
from utils.plotting_utils import plot
from utils.loss_functions import get_feature_loss_function, calculate_neuron_activations, calculate_maha_dists, compute_losses, get_loss_function, calculate_confs
from utils.inet_classes import IDX2NAME
from utils.parser_utils import CommonArguments

from utils.attribution_with_gradient import ActivationCaption
import numpy as np


# skincancer
CLASSES=['nontumor_skin_dermis_dermis',
 'nontumor_skin_epidermis_epidermis',
 'tumor_skin_naevus_naevus',
 'tumor_skin_melanoma_melanoma',
 'nontumor_skin_subcutis_subcutis',
 'nontumor_skin_sebaceousglands_sebaceousglands',
 'tumor_skin_epithelial_bcc',
 'tumor_skin_epithelial_sqcc',
 'nontumor_skin_muscle_skeletal',
 'nontumor_skin_chondraltissue_chondraltissue',
 'nontumor_skin_sweatglands_sweatglands',
 'nontumor_skin_necrosis_necrosis',
 'nontumor_skin_hairfollicle_hairfollicle',
 'nontumor_skin_nerves_nerves',
 'nontumor_skin_vessel_vessel',
 'nontumor_skin_elastosis_elastosis']

TAIL_CLASSES=['nontumor_skin_dermis_dermis',
 'nontumor_skin_epidermis_epidermis',
 'tumor_skin_naevus_naevus',
 'nontumor_skin_subcutis_subcutis',
 'nontumor_skin_sebaceousglands_sebaceousglands',
 'tumor_skin_epithelial_bcc',
 'nontumor_skin_muscle_skeletal',
 'nontumor_skin_chondraltissue_chondraltissue',
 'nontumor_skin_sweatglands_sweatglands',
 'nontumor_skin_necrosis_necrosis',
 'nontumor_skin_vessel_vessel',
 'nontumor_skin_elastosis_elastosis']
HEAD_CLASSES=[c for c in CLASSES if c not in TAIL_CLASSES]
MED_CLASSES=[]

head_med_idxs=[i for i,c in enumerate(CLASSES) if c not in TAIL_CLASSES]

tail_idxs=[i for i,c in enumerate(CLASSES) if c in TAIL_CLASSES]
@dataclass
class GuidedArgs(CommonArguments):
    gpu: int = 0
    seed: int = 42
    num_images: int = 1
    local_split: int = 0
    num_splits: int = 0

    target_classes: Optional[List[int]] = None
    target_neurons: Optional[List[int]] = None
    spurious_neurons: bool = False
    min_head_med: bool = False

    early_stopping_confidence: float = 1.0

    results_folder: str = 'output_cvpr/imagenet_guided'

    solver: str = 'ddim'

    regularizers: List[str] = field(default_factory=lambda: [])
    regularizers_ws: List[float] = field(default_factory=lambda: [])
    segmentation_squash: Optional[str] = None

    loss: str = 'neuron_activation_plus_log_confidence'

    prompt: str = 'spagetti bolognese'
    all_classes: bool = False

    weight_path: str = '/path/to/lora/weights'
    model_path: str = '/path/to/aux/classifier/weights'

def setup() -> GuidedArgs:
    default_config: GuidedArgs = OmegaConf.structured(GuidedArgs)
    cli_args = OmegaConf.from_cli()
    config: GuidedArgs = OmegaConf.merge(default_config, cli_args)
    return config


if __name__=="__main__":
    args = setup()
    print(args)
    device = torch.device(f'cuda:{args.gpu}')
    
    if args.spurious_neurons:
        target_classes_neurons = torch.load('src/utils/spurious_class_neuron_pairs.pth')
    else:
        target_classes = args.target_classes
        target_neurons = args.target_neurons
        assert len(target_classes) == len(target_neurons)
        target_classes_neurons = zip(target_classes, target_neurons)
    if args.all_classes:
        target_classes=[i for i in range(1000)]
        target_neurons=[0 for i in range(1000)]
        target_classes_neurons = zip(target_classes, target_neurons)

    if args.min_head_med:
        target_classes = head_med_idxs
        target_neurons=[0 for i in (head_med_idxs)]
        target_classes_neurons = zip(target_classes, target_neurons)
    loss = args.loss
    print(target_classes, target_neurons)
    model_path = args.model_path #"/scratch/mmueller67/CVPR_ppdg/diffusers/examples/text_to_image/skincancer-similar/"
    pipe = StableDiffusionPipelineWithGrad.from_pretrained("CompVis/stable-diffusion-v1-4", #torch_dtype=torch.float16,
                    safety_checker=None)
    # pipe.unet.load_attn_procs(model_path)
    pipe.load_lora_weights(model_path, weight_name="pytorch_lora_weights.safetensors")
    pipe.fuse_lora()
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

    num_cutouts = args.augmentation_num_cutouts
    
    weight_path = args.weight_path # '/scratch/mmueller67/LongTailCXR/skincancer_results/Skin_dataset_similar_PIL_resnet50_rw-sklearn_ldam-drw_lr-0.0001_bs-256/chkpt_epoch-33.pt'
    classifier = load_tv(weight_path, N_CLASSES=16)
    classifier.to(device)

    classifier_cam = None
    classifier_cam_target_layers = None

    result_folder_pre = f'{args.results_folder}'
    augm_string = f'_noise_{args.augmentation_noise_schedule}_{args.augmentation_noise_sd}'


    early_stopping_loss = -args.early_stopping_confidence

    print(f'Writing results to: {result_folder_pre}')


    loss_function = get_loss_function(loss, classifier)
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
    torch.manual_seed(99) # for validation set #torch.manual_seed(0)
    num_classes = 16
    # deterministic_latents = torch.randn((num_classes, args.num_images if args.num_images < 50 else 50) + latent_dim, dtype=torch.float)
    deterministic_latents = torch.randn((num_classes, args.num_images) + latent_dim, dtype=torch.float)

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
    output_description +='earlyS'+str(args.early_stopping_confidence)

    to_pil = transforms.ToPILImage()

    result_folder = os.path.join(result_folder_pre, output_description)
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, 'config.yaml'), "w") as f:
        OmegaConf.save(args, f)
    
    target_i, target_class, target_neuron = 0,0,0
    target_class=target_classes[0]
    for i in [0]:
        act_dict = {'sd':[], 'ours':[]}
        for img_idx in range(0,args.num_images):
            
            if args.num_splits > 1:
                current_split_idx = target_i % args.num_splits
                if not current_split_idx == args.local_split:
                    continue

            target_label = CLASSES[target_class]
            if args.min_head_med:
                target_label = 'Min head med'
            prompt = args.prompt
            target = target_classes
            result_folder_postfix = f'{prompt}_{target_label}'
            result_folder = os.path.join(result_folder_pre, output_description, result_folder_postfix)
            out_file_pth = os.path.join(result_folder, f'{img_idx}.pdf')
            
            caption = f'Prompt: {prompt}'
            
            os.makedirs(result_folder, exist_ok=True)

            target_preds = torch.zeros(args.num_images, dtype=torch.long)
            
            latent = deterministic_latents[target_class, img_idx][None,:] 
            null_texts_embeddings = None
            target_prompt_str = None
            original_tensor = None
            reconstructed_tensor = None

            prompt_str = f'{prompt}'

            return_values = pipe(target,
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
            loss_scores = return_values['loss_scores']
            regularizer_scores = return_values['regularizer_scores']


            img_grid=torch.clamp(img_grid,0,1) # clamp already before losses are computed
            confs, preds = calculate_confs(classifier, img_grid, device, target_class=None, return_predictions=True)
            preds_to_plot = [CLASSES[pred]+f' ({pred})' for pred in preds]
            confs_head_med = calculate_confs(classifier, img_grid, device, target_class=head_med_idxs, return_predictions=False)
            confs_agg=torch.zeros(img_grid.shape[0],len(target_classes))
            for i_,target_class in enumerate(target_classes):
                print(target_class)
                confs_target, preds_target, entropies = calculate_confs(classifier, img_grid, device, target_class=[target_class], return_predictions=True, return_entropy=True)
                confs_agg[:,i_]=confs_target
            print(confs_agg)
            confs_target=confs_agg.sum(-1)
            losses = -confs_target#confs
            print(confs.shape)
            print('confs: ', confs)
            max_act_idx = torch.argmax(torch.FloatTensor(confs_target)).item() # use max conf of target class as selection criterion
            print('confs: ',confs_target)
            print(max_act_idx)
            print('conf max act: ', max_act_idx, confs_target[max_act_idx])
            max_act_img = img_grid[max_act_idx]

            act_dict['sd'].append(losses[0])
            act_dict['ours'].append(losses[max_act_idx])

            pil_sd = to_pil(torch.clamp(img_grid[0], 0, 1))
            pil_sd = pil_sd.save(os.path.join(result_folder, f'{img_idx}_sd.png'))
            pil_ours = to_pil(torch.clamp(max_act_img, 0, 1))
            pil_ours = pil_ours.save(os.path.join(result_folder, f'{img_idx}_ours.png'))

            plot(original_tensor, reconstructed_tensor, img_grid, 'Max conf head-med', confs_head_med,
                 os.path.join(result_folder, f'{img_idx}.pdf'), loss_scores=loss_scores,
                 regularizer_scores=regularizer_scores, preds=preds_to_plot, confs=confs)