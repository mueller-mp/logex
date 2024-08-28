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
from utils.sd_backprop_pipe import StableDiffusionPipelineWithGrad
from utils.plotting_utils import plot
from utils.loss_functions import get_feature_loss_function, calculate_neuron_activations, calculate_maha_dists, compute_losses
from utils.inet_classes import IDX2NAME
from utils.parser_utils import CommonArguments

from utils.attribution_with_gradient import ActivationCaption
import numpy as np

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

    early_stopping_confidence: float = 1.0

    results_folder: str = 'output_cvpr/imagenet_guided'

    solver: str = 'ddim'

    regularizers: List[str] = field(default_factory=lambda: [])
    regularizers_ws: List[float] = field(default_factory=lambda: [])
    segmentation_squash: Optional[str] = None

    loss: str = 'neuron_activation_plus_log_confidence'

    prompt: str = 'spagetti bolognese'
    all_classes: bool = False

def setup() -> GuidedArgs:
    default_config: GuidedArgs = OmegaConf.structured(GuidedArgs)
    cli_args = OmegaConf.from_cli()
    config: GuidedArgs = OmegaConf.merge(default_config, cli_args)
    return config


# X python src/imagenet_neuron_activation.py gpu=0 spurious_neurons=True num_splits=3 local_split=0 loss=neuron_activation_plus_log_confidence
# X python src/imagenet_neuron_activation.py gpu=2 spurious_neurons=True num_splits=3 local_split=1 loss=neuron_activation_plus_log_confidence
# X python src/imagenet_neuron_activation.py gpu=4 spurious_neurons=True num_splits=3 local_split=2 loss=neuron_activation_plus_log_confidence

# X python src/imagenet_neuron_activation.py gpu=1 spurious_neurons=True num_splits=3 local_split=0 loss=neg_neuron_activation_plus_log_confidence
# X python src/imagenet_neuron_activation.py gpu=3 spurious_neurons=True num_splits=3 local_split=1 loss=neg_neuron_activation_plus_log_confidence
# X python src/imagenet_neuron_activation.py gpu=5 spurious_neurons=True num_splits=3 local_split=2 loss=neg_neuron_activation_plus_log_confidence

# python src/imagenet_neuron_activation.py gpu=0 target_classes=[2] target_neurons=[1697] loss=neuron_activation_plus_log_confidence
# python src/imagenet_neuron_activation.py gpu=1 target_classes=[2] target_neurons=[1697] loss=neg_neuron_activation_plus_log_confidence

# for Mahalanobis
# python src/imagenet_neuron_activation.py gpu=0 loss=max_single_maha_score_targeted target_neurons=[674] target_classes=[385] sgd_steps=20 prompt="a book"       

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

    num_cutouts = args.augmentation_num_cutouts
    
    # classifier = load_madry_l2_with_cutout(cut_power=0.3, num_cutouts=num_cutouts, noise_sd=args.augmentation_noise_sd, 
    #                                             noise_schedule=args.augmentation_noise_schedule, 
    #                                             noise_descending_steps=args.sgd_steps)
    # model_name1 = 'B_32-i1k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224'
    model_name1 = 'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224'
    classifier = load_vit_with_cutout(model_name1,cut_power=0.3, num_cutouts=num_cutouts, noise_sd=args.augmentation_noise_sd, 
                                                noise_schedule=args.augmentation_noise_schedule, 
                                                noise_descending_steps=args.sgd_steps)
    # classifier = load_vit(model_name1,cut_power=0.3, num_cutouts=num_cutouts, noise_sd=args.augmentation_noise_sd, 
    #                                             noise_schedule=args.augmentation_noise_schedule, 
    #                                             noise_descending_steps=args.sgd_steps)
        
    last_layer = classifier.model.head     # classifier.model.fc
    classifier.to(device)

    layer_activations = ActivationCaption(classifier, [last_layer])

    path=f'/mnt/SHARED/mmueller67/ood_tests/NINCO/cache/cache_methods/{model_name1}'
    mean_path1 = os.path.join(path, 'mean.npy')
    prec_path1 = os.path.join(path, 'prec.npy')
    mean1 = np.load(mean_path1, allow_pickle=True)
    prec1 = np.load(prec_path1, allow_pickle=True)
    mean1 = torch.from_numpy(mean1).to(device).double()
    prec1 = torch.from_numpy(prec1).to(device).double()

    # threshold for ID/OOD for FPR@TPR95
    maha_id_scores = np.load(f'/mnt/SHARED/mmueller67/ood_tests/NINCO/cache/cache_methods/{model_name1}/maha_id_scores.npy')
    t1_maha = np.quantile(maha_id_scores, (1 - 0.95))

    logits = np.load(f'/mnt/SHARED/mmueller67/ood_tests/NINCO/cache/cache_val/{model_name1}/logits_0.npy')
    softmax = torch.softmax(torch.from_numpy(logits).to(device),-1)
    max_softmax = softmax.max(-1)[0].cpu().numpy()
    t1_msp = np.quantile(max_softmax, 1-0.95)

    # second model 
    model_name2 = "B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224"
    #'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224'
    classifier2 = load_vit_with_cutout(model_name2,cut_power=0.3, num_cutouts=num_cutouts, noise_sd=args.augmentation_noise_sd, 
                                                noise_schedule=args.augmentation_noise_schedule, 
                                                noise_descending_steps=args.sgd_steps)
    # classifier2 = load_vit(model_name2,cut_power=0.3, num_cutouts=num_cutouts, noise_sd=args.augmentation_noise_sd, 
    #                                             noise_schedule=args.augmentation_noise_schedule, 
    #                                             noise_descending_steps=args.sgd_steps)
    
    last_layer2 = classifier2.model.head     # classifier.model.fc
    classifier2.to(device)

    layer_activations2 = ActivationCaption(classifier2, [last_layer2])

    path=f'/mnt/SHARED/mmueller67/ood_tests/NINCO/cache/cache_methods/{model_name2}'
    mean_path2 = os.path.join(path, 'mean.npy')
    prec_path2 = os.path.join(path, 'prec.npy')
    mean2 = np.load(mean_path2, allow_pickle=True)
    prec2 = np.load(prec_path2, allow_pickle=True)
    mean2 = torch.from_numpy(mean2).to(device).double()
    prec2 = torch.from_numpy(prec2).to(device).double()

    maha_id_scores = np.load(f'/mnt/SHARED/mmueller67/ood_tests/NINCO/cache/cache_methods/{model_name2}/maha_id_scores.npy')
    t2_maha = np.quantile(maha_id_scores, (1 - 0.95))

    logits = np.load(f'/mnt/SHARED/mmueller67/ood_tests/NINCO/cache/cache_val/{model_name2}/logits_0.npy')
    softmax = torch.softmax(torch.from_numpy(logits).to(device),-1)
    max_softmax = softmax.max(-1)[0].cpu().numpy()
    t2_msp = np.quantile(max_softmax, 1-0.95)

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=248, interpolation=TF.InterpolationMode.BICUBIC, max_size=None, antialias=None),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
   # torchvision.transforms.Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000]), std=torch.tensor([0.5000, 0.5000, 0.5000]))
    ])

    classifier_cam = None
    classifier_cam_target_layers = None

    result_folder_pre = f'{args.results_folder}_maha_neg_dists'
    augm_string = f'_noise_{args.augmentation_noise_schedule}_{args.augmentation_noise_sd}'


    early_stopping_loss = None#-args.early_stopping_confidence

    print(f'Writing results to: {result_folder_pre}')


    loss_function = get_feature_loss_function(loss, classifier, layer_activations, classifier2=classifier2, layer_activations2=layer_activations2, mean1=mean1, prec1=prec1, mean2=mean2, prec2=prec2)
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
    
    for target_i, (target_class, target_neuron) in enumerate(target_classes_neurons):
    # target_i, target_class, target_neuron = 0,0,0
    # for i in [0]:
        act_dict = {'sd':[], 'ours':[]}
        for img_idx in range(args.num_images):
            
            if args.num_splits > 1:
                current_split_idx = target_i % args.num_splits
                if not current_split_idx == args.local_split:
                    continue

            target_label = IDX2NAME[target_class]
            prompt = args.prompt
            target = [target_class, target_neuron]
            # result_folder_postfix = f'{target_class}_{target_label}_neuron_{target_neuron}'
            result_folder_postfix = f'{prompt}_{target_class}_{target_label}'
            result_folder = os.path.join(result_folder_pre, output_description, result_folder_postfix)
            out_file_pth = os.path.join(result_folder, f'{img_idx}.pdf')
            
            # caption = f'Neuron {target_neuron}'
            caption = f'Prompt: {prompt}'
            if os.path.isfile(out_file_pth):
                continue
            
            os.makedirs(result_folder, exist_ok=True)

            target_preds = torch.zeros(args.num_images, dtype=torch.long)
            
            latent = deterministic_latents[target_class, img_idx][None,:]
            null_texts_embeddings = None
            target_prompt_str = None
            original_tensor = None
            reconstructed_tensor = None

            prompt_str = f'a photograph of a {prompt}'
            # prompt_str = f'a black photograph'

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

            # losses = calculate_neuron_activations(classifier, layer_activations, img_grid, device, target, loss)                  
            neg_dists_maha, preds_maha, confs, preds, neg_dists_maha_raw, neg_dists_maha_target, neg_dists_maha_target_raw, target_confs = calculate_maha_dists(layer_activations, img_grid, device, mean1, prec1, transform, layer_activations2, mean2, prec2, classifier1=classifier,classifier2=classifier2,target_class=target_class)                  

            target_preds = np.array([[IDX2NAME[p.item()] for p in pred] for pred in preds])
            target_preds_maha = np.array([[IDX2NAME[p.item()] for p in pred] for pred in preds_maha])
            confs = confs.cpu().numpy()
            target_confs = target_confs.cpu().numpy()
            # select cleverly
            losses = compute_losses(args.loss, neg_dists_maha, neg_dists_maha_target, confs, target_confs)
            max_act_idx = torch.argmin(torch.FloatTensor(losses)).item()

            max_act_img = img_grid[max_act_idx]
            # sd_file_pth = os.path.join(result_folder, f'{img_idx}_sd.pth')
            # torch.save(img_grid[0], sd_file_pth)
            # out_file_pth = os.path.join(result_folder, f'{img_idx}_ours.pth')
            # torch.save(max_act_img, out_file_pth)

            act_dict['sd'].append(losses[0])
            act_dict['ours'].append(losses[max_act_idx])

            pil_sd = to_pil(torch.clamp(img_grid[0], 0, 1))
            pil_sd = pil_sd.save(os.path.join(result_folder, f'{img_idx}_sd.png'))
            pil_ours = to_pil(torch.clamp(max_act_img, 0, 1))
            pil_ours = pil_ours.save(os.path.join(result_folder, f'{img_idx}_ours.png'))

            plot(None, None, img_grid, caption, neg_dists_maha, os.path.join(result_folder, f'{img_idx}.pdf'),attribute_values_over_trajectory_2=neg_dists_maha_raw,preds=target_preds, preds_maha=target_preds_maha,confs=confs,t1_maha=t1_maha, t2_maha=t2_maha, t1_msp=t1_msp, t2_msp=t2_msp, loss_scores=losses)
        
        # act_file_pth  = os.path.join(result_folder, f'target_activations.pth')
        # torch.save(act_file_pth, act_file_pth)