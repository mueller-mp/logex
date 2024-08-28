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


# nih
# CLASSES =['No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule',
#        'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening',
#        'Cardiomegaly', 'Fibrosis', 'Edema', 'Tortuous Aorta', 'Emphysema',
#        'Pneumonia', 'Calcification of the Aorta', 'Pneumoperitoneum', 'Hernia',
#        'Subcutaneous Emphysema', 'Pneumomediastinum'] # df_bal_test_true.columns
# TAIL_CLASSES = ['Pneumomediastinum','Subcutaneous Emphysema','Hernia','Pneumoperitoneum']
# HEAD_CLASSES = ['Pneumothorax',
#  'Mass',
#  'Nodule',
#  'Effusion',
#  'Atelectasis',
#  'Infiltration',
#  'No Finding']
# MED_CLASSES = ['Calcification of the Aorta',
#  'Pneumonia',
#  'Emphysema',
#  'Tortuous Aorta',
#  'Edema',
#  'Fibrosis',
#  'Cardiomegaly',
#  'Pleural_Thickening',
#  'Consolidation']

# mimic
# CLASSES = [
#             'No Finding', 'Lung Opacity', 'Cardiomegaly', 'Atelectasis',
#             'Pleural Effusion', 'Support Devices', 'Edema', 'Pneumonia',
#             'Pneumothorax', 'Lung Lesion', 'Fracture', 'Enlarged Cardiomediastinum',
#             'Consolidation', 'Pleural Other', 'Calcification of the Aorta',
#             'Tortuous Aorta', 'Pneumoperitoneum', 'Subcutaneous Emphysema',
#             'Pneumomediastinum',
#         ]
# TAIL_CLASSES = ['Pneumoperitoneum', 'Subcutaneous Emphysema', 'Pneumomediastinum']
# MED_CLASSES = ['Fracture',
#  'Enlarged Cardiomediastinum',
#  'Consolidation',
#  'Pleural Other',
#  'Calcification of the Aorta',
#  'Tortuous Aorta']
# HEAD_CLASSES = ['No Finding',
#  'Lung Opacity',
#  'Cardiomegaly',
#  'Atelectasis',
#  'Pleural Effusion',
#  'Support Devices',
#  'Edema',
#  'Pneumonia',
#  'Pneumothorax',
#  'Lung Lesion']

# isic
# CLASSES=['NV', 'MEL', 'BCC', 'BKL', 'AK', 'SCC', 'DF', 'VASC']
# TAIL_CLASSES=['SCC', 'DF', 'VASC']
# MED_CLASSES=[ 'BKL', 'AK']
# HEAD_CLASSES=['NV', 'MEL', 'BCC']

# very very lt skincancer
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
# MED_CLASSES=[ 'nontumor_skin_subcutis_subcutis',
#  'nontumor_skin_sebaceousglands_sebaceousglands',
#  'tumor_skin_epithelial_bcc',
#  'tumor_skin_epithelial_sqcc']
# HEAD_CLASSES=['nontumor_skin_dermis_dermis',
#  'nontumor_skin_epidermis_epidermis',
#  'tumor_skin_naevus_naevus',
#  'tumor_skin_melanoma_melanoma']
# TAIL_CLASSES=['nontumor_skin_muscle_skeletal',
#  'nontumor_skin_chondraltissue_chondraltissue',
#  'nontumor_skin_sweatglands_sweatglands',
#  'nontumor_skin_necrosis_necrosis',
#  'nontumor_skin_hairfollicle_hairfollicle',
#  'nontumor_skin_nerves_nerves',
#  'nontumor_skin_vessel_vessel',
#  'nontumor_skin_elastosis_elastosis']

# similar dset

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

# for guided CXR generation, max tail probability
# python src/cxr_classifier_conf.py gpu=6 loss=conf target_neurons=[674] target_classes=[1] sgd_steps=20 optim=adam sgd_stepsize=0.01 sgd_schedule=none latent_lr_factor=0.1 optimize_latents=True optimize_conditioning=True optimize_uncond=True loss_steps=3 loss_steps_schedule=linear per_timestep_conditioning=True per_timestep_uncond=True augmentation_num_cutouts=16 augmentation_noise_sd=0.005 augmentation_noise_schedule=const regularizers=[latent_background_l2,px_background_l2,px_background_lpips] regularizers_ws=[25.0,250.0,25.0] segmentation_squash=sqrt_0.3 prompt="A front-view chest X-ray image with diagnosis Pneumoperitoneum" min_head_med=True num_images=1000 results_folder='nih_guided'
# for guided CXR generation, max tail probability, max specific tail class - have to adapt target_class and prompt
# 16 'Pneumoperitoneum',
# 17 'Hernia',
# 18 'Subcutaneous Emphysema',
# 19 'Pneumomediastinum'
# python src/cxr_classifier_conf.py gpu=4 loss=conf target_neurons=[674] target_classes=[19] sgd_steps=20 optim=adam sgd_stepsize=0.01 sgd_schedule=none latent_lr_factor=0.1 optimize_latents=True optimize_conditioning=True optimize_uncond=True loss_steps=3 loss_steps_schedule=linear per_timestep_conditioning=True per_timestep_uncond=True augmentation_num_cutouts=16 augmentation_noise_sd=0.005 augmentation_noise_schedule=const regularizers=[latent_background_l2,px_background_l2,px_background_lpips] regularizers_ws=[25.0,250.0,25.0] segmentation_squash=sqrt_0.3 prompt="A front-view chest X-ray image with diagnosis Pneumomediastinum" num_images=1000 results_folder='nih_guided_class_specific'
# with MIMIC:
#['Pneumoperitoneum' 16, 'Subcutaneous Emphysema' 17, 'Pneumomediastinum' 18]
# 
# python src/cxr_classifier_conf.py gpu=3 loss=conf target_neurons=[674] target_classes=[18] sgd_steps=20 optim=adam sgd_stepsize=0.01 sgd_schedule=none latent_lr_factor=0.1 optimize_latents=True optimize_conditioning=True optimize_uncond=True loss_steps=3 loss_steps_schedule=linear per_timestep_conditioning=True per_timestep_uncond=True augmentation_num_cutouts=16 augmentation_noise_sd=0.005 augmentation_noise_schedule=const regularizers=[latent_background_l2,px_background_l2,px_background_lpips] regularizers_ws=[25.0,250.0,25.0] segmentation_squash=sqrt_0.3 prompt="A front-view chest X-ray image with diagnosis Pneumomediastinum" num_images=1000 results_folder='mimic_guided_class_specific'
# ISIC
# 5 Squamous cell carcinoma, 6 Dermatofibroma, 7 Vascular lesion,
# python src/cxr_classifier_conf.py gpu=6 loss=conf target_neurons=[674] target_classes=[7] sgd_steps=20 optim=adam sgd_stepsize=0.01 sgd_schedule=none latent_lr_factor=0.1 optimize_latents=True optimize_conditioning=True optimize_uncond=True loss_steps=3 loss_steps_schedule=linear per_timestep_conditioning=True per_timestep_uncond=True augmentation_num_cutouts=16 augmentation_noise_sd=0.005 augmentation_noise_schedule=const regularizers=[latent_background_l2,px_background_l2,px_background_lpips] regularizers_ws=[25.0,250.0,25.0] segmentation_squash=sqrt_0.3 prompt="A dermoscopic image with diagnosis Vascular lesion" num_images=1000 results_folder='isic_guided_class_specific'

# skincancer
# 8 A histopathological slide from a patient with skeletal muscle
# 9 A histopathological slide from a patient with chrondal tissue
# 10 A histopathological slide from a patient with sweat glands
# 11 A histopathological slide from a patient with necrosis
# 12 A histopathological slide from a patient with hair follicle
# 13 A histopathological slide from a patient with nerves
# 14 A histopathological slide from a patient with vessels
# 15 A histopathological slide from a patient with elastosis
# python src/cxr_classifier_conf.py gpu=4 loss=conf target_neurons=[674] target_classes=[8] sgd_steps=20 optim=adam sgd_stepsize=0.01 sgd_schedule=none latent_lr_factor=0.1 optimize_latents=True optimize_conditioning=True optimize_uncond=True loss_steps=3 loss_steps_schedule=linear per_timestep_conditioning=True per_timestep_uncond=True augmentation_num_cutouts=16 augmentation_noise_sd=0.005 augmentation_noise_schedule=const regularizers=[latent_background_l2,px_background_l2,px_background_lpips] regularizers_ws=[25.0,250.0,25.0] segmentation_squash=sqrt_0.3 prompt="A histopathological slide from a patient with skeletal muscle" num_images=10 results_folder='skincancer_vvlt_guided_class_specific_selectmaxconf'


# skincancer similar
# 0 A histopathological slide from a patient with dermis
# 1 A histopathological slide from a patient with epidermis
# 2 A histopathological slide from a patient with naevus
# 4 A histopathological slide from a patient with subcutis
# 5 A histopathological slide from a patient with sebaceousglands
# 6 A histopathological slide from a patient with basal cell carcinoma
# 8 A histopathological slide from a patient with skeletal muscle
# 9 A histopathological slide from a patient with chrondal tissue
# 10 A histopathological slide from a patient with sweat glands
# 11 A histopathological slide from a patient with necrosis
# 14 A histopathological slide from a patient with vessels
# 15 A histopathological slide from a patient with elastosis
# python src/cxr_classifier_conf.py gpu=4 loss=conf target_neurons=[674] target_classes=[0] sgd_steps=20 optim=adam sgd_stepsize=0.01 sgd_schedule=none latent_lr_factor=0.1 optimize_latents=True optimize_conditioning=True optimize_uncond=True loss_steps=3 loss_steps_schedule=linear per_timestep_conditioning=True per_timestep_uncond=True augmentation_num_cutouts=16 augmentation_noise_sd=0.005 augmentation_noise_schedule=const regularizers=[latent_background_l2,px_background_l2,px_background_lpips] regularizers_ws=[25.0,250.0,25.0] segmentation_squash=sqrt_0.3 prompt="A histopathological slide from a patient with dermis" num_images=10 results_folder='skincancer_similar_guided_class_specific_selectmaxconf'
# python src/cxr_classifier_conf.py gpu=4 loss=neg_entropy target_neurons=[674] target_classes=[0] sgd_steps=20 optim=adam sgd_stepsize=0.01 sgd_schedule=none latent_lr_factor=0.1 optimize_latents=True optimize_conditioning=True optimize_uncond=True loss_steps=3 loss_steps_schedule=linear per_timestep_conditioning=True per_timestep_uncond=True augmentation_num_cutouts=16 augmentation_noise_sd=0.005 augmentation_noise_schedule=const regularizers=[latent_background_l2,px_background_l2,px_background_lpips] regularizers_ws=[25.0,250.0,25.0] segmentation_squash=sqrt_0.3 prompt="A histopathological slide from a patient with dermis" num_images=10 results_folder='skincancer_similar_guided_class_specific_entropy'

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
    # pipe = StableDiffusionPipelineWithGrad.from_pretrained(args.model_path)
    # model_path = "/scratch/mmueller67/diffusers/examples/text_to_image/cxr8-lt-lora/"
    # model_path = "/scratch/mmueller67/diffusers/examples/text_to_image/mimic-lt-tail-lora/"
    # model_path = "/scratch/mmueller67/diffusers/examples/text_to_image/skincancer-vvlt-tail-lora/"
    model_path = "/scratch/mmueller67/diffusers/examples/text_to_image/skincancer-similar-tail-lora/"
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
    
    # import torchvision
    # classifier = torchvision.models.resnet50()
    # classifier.fc = torch.nn.Linear(classifier.fc.in_features, 20)
    # weight_path = '/scratch/mmueller67/LongTailCXR/nih_results_decoupling/nih-cxr-lt_resnet50_decoupling-cRT_ce_lr-0.0001_bs-256_from-focal_fl-gamma-2.0/chkpt_epoch-9.pt'
    # weight_path = '/scratch/mmueller67/LongTailCXR/mimic_results/mimic-cxr-lt_resnet50_rw-sklearn_ldam-drw_lr-0.0001_bs-256/chkpt_epoch-21.pt'
    # weight_path = '/scratch/mmueller67/LongTailCXR/isic_results/isic2019_lt_resnet50_rw-sklearn_ldam-drw_lr-0.0001_bs-256/chkpt_epoch-19.pt'
    # weight_path = '/scratch/mmueller67/LongTailCXR/skincancer_results/Skin_dataset_very_very_lt_resnet50_rw-sklearn_ldam-drw_lr-0.0001_bs-256/chkpt_epoch-37.pt'
    # weight path without lt
    # weight_path = '/scratch/mmueller67/LongTailCXR/skincancer_results/Skin_dataset_resnet50_rw-sklearn_ce_lr-0.0001_bs-256/chkpt_epoch-28.pt'
    # weight_path = '/scratch/mmueller67/LongTailCXR/skincancer_results/Skin_dataset_similar_resnet50_rw-sklearn_ldam-drw_lr-0.0001_bs-256/chkpt_epoch-29.pt'
    # weight_path='/scratch/mmueller67/LongTailCXR/skincancer_results/Skin_dataset_resnet50_rw-cb_ce_cb-beta-0.9999_lr-0.0001_bs-256/chkpt_epoch-13.pt' # weights were not used for any classifier guidance
    weight_path = '/scratch/mmueller67/LongTailCXR/skincancer_results/Skin_dataset_similar_PIL_resnet50_rw-sklearn_ldam-drw_lr-0.0001_bs-256/chkpt_epoch-33.pt'
    # weight_path = '/scratch/mmueller67/LongTailCXR/skincancer_results/Skin_dataset_PIL_resnet50_rw-sklearn_ldam-drw_lr-0.0001_bs-256/chkpt_epoch-11.pt'
    classifier = load_tv(weight_path, N_CLASSES=16)
    classifier.to(device)

    classifier_cam = None
    classifier_cam_target_layers = None

    result_folder_pre = f'{args.results_folder}'
    augm_string = f'_noise_{args.augmentation_noise_schedule}_{args.augmentation_noise_sd}'


    early_stopping_loss = -args.early_stopping_confidence#-0.4# None #0.005# None#  -args.early_stopping_confidence

    print(f'Writing results to: {result_folder_pre}')


    # loss_function = get_feature_loss_function(loss, classifier, layer_activations, classifier2=classifier2, layer_activations2=layer_activations2, mean1=mean1, prec1=prec1, mean2=mean2, prec2=prec2)
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
    torch.manual_seed(0)
    num_classes = 20
    # deterministic_latents = torch.randn((num_classes, args.num_images if args.num_images < 50 else 50) + latent_dim, dtype=torch.float)
    deterministic_latents = torch.randn((num_classes, args.num_images) + latent_dim, dtype=torch.float)
    # What's this??

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
    
    # for target_i, (target_class) in enumerate(target_classes):
    target_i, target_class, target_neuron = 0,0,0
    for i in [0]:
        act_dict = {'sd':[], 'ours':[]}
        for img_idx in range(0,args.num_images):
            
            if args.num_splits > 1:
                current_split_idx = target_i % args.num_splits
                if not current_split_idx == args.local_split:
                    continue

            # target_label = IDX2NAME[target_class]
            target_label = CLASSES[target_class]
            if args.min_head_med:
                target_label = 'Min head med'
            prompt = args.prompt
            # target = [target_class, target_neuron]
            # target = [target_class]
            target = target_classes
            # result_folder_postfix = f'{target_class}_{target_label}_neuron_{target_neuron}'
            result_folder_postfix = f'{prompt}_{target_label}'
            # result_folder_postfix = f'{prompt}_{target_class}_{target_label}'
            result_folder = os.path.join(result_folder_pre, output_description, result_folder_postfix)
            out_file_pth = os.path.join(result_folder, f'{img_idx}.pdf')
            
            # caption = f'Neuron {target_neuron}'
            caption = f'Prompt: {prompt}'
            # if os.path.isfile(out_file_pth):
            #     continue
            
            os.makedirs(result_folder, exist_ok=True)

            target_preds = torch.zeros(args.num_images, dtype=torch.long)
            
            latent = deterministic_latents[target_class, img_idx][None,:] # What's this??
            null_texts_embeddings = None
            target_prompt_str = None
            original_tensor = None
            reconstructed_tensor = None

            prompt_str = f'{prompt}'
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


            img_grid=torch.clamp(img_grid,0,1) # clamp already before losses are computed

            # losses = calculate_neuron_activations(classifier, layer_activations, img_grid, device, target, loss)                  
            # neg_dists_maha, preds_maha, confs, preds, neg_dists_maha_raw, neg_dists_maha_target, neg_dists_maha_target_raw, target_confs = calculate_maha_dists(layer_activations, img_grid, device, mean1, prec1, transform, layer_activations2, mean2, prec2, classifier1=classifier,classifier2=classifier2,target_class=target_class)                  

            # target_preds = np.array([[IDX2NAME[p.item()] for p in pred] for pred in preds])
            # target_preds_maha = np.array([[IDX2NAME[p.item()] for p in pred] for pred in preds_maha])
            # confs = confs.cpu().numpy()
            # target_confs = target_confs.cpu().numpy()
            # # select cleverly
            # losses = compute_losses(args.loss, neg_dists_maha, neg_dists_maha_target, confs, target_confs)
            confs, preds = calculate_confs(classifier, img_grid, device, target_class=None, return_predictions=True)
            preds_to_plot = [CLASSES[pred]+f' ({pred})' for pred in preds]
            confs_head_med = calculate_confs(classifier, img_grid, device, target_class=head_med_idxs, return_predictions=False)
            confs_target, preds_target, entropies = calculate_confs(classifier, img_grid, device, target_class=target_classes, return_predictions=True, return_entropy=True)
            # confs_target, preds_target, entropies
            # losses = compute_losses(args.loss, None, None, confs, target_confs=None)
            losses = entropies#-confs
            print(confs.shape)
            # max_act_idx = torch.argmin(torch.FloatTensor(confs_head_med)).item() # use min head-med conf as selection criterion
            max_act_idx = torch.argmax(torch.FloatTensor(confs_target)).item() # use max conf of target class as selection criterion
            print('confs: ',confs_target)
            print('conf max act: ', max_act_idx, confs_target[max_act_idx])
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

            # plot(None, None, img_grid, caption, neg_dists_maha, os.path.join(result_folder, f'{img_idx}.pdf'),attribute_values_over_trajectory_2=neg_dists_maha_raw,preds=target_preds, preds_maha=target_preds_maha,confs=confs,t1_maha=t1_maha, t2_maha=t2_maha, t1_msp=t1_msp, t2_msp=t2_msp, loss_scores=losses)
            plot(original_tensor, reconstructed_tensor, img_grid, 'Max conf head-med', confs_head_med,
                 os.path.join(result_folder, f'{img_idx}.pdf'), loss_scores=loss_scores,
                 regularizer_scores=regularizer_scores, preds=preds_to_plot, confs=confs)
        # act_file_pth  = os.path.join(result_folder, f'target_activations.pth')
        # torch.save(act_file_pth, act_file_pth)