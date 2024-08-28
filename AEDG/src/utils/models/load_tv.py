import torch
import torchvision.models as torch_models
import os
from .model_wrappers import NormalizationAndResizeWrapper, NormalizationAndCutoutWrapper,\
    IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
# WEIGHT_PATH='/scratch/mmueller67/LongTailCXR/nih_results_decoupling/nih-cxr-lt_resnet50_decoupling-cRT_ce_lr-0.0001_bs-256_from-focal_fl-gamma-2.0/chkpt_epoch-9.pt'

def load_tv_with_cutout(wight_path,cut_power, num_cutouts, noise_sd=0, noise_schedule='constant',
                                noise_descending_steps=None,
                                batches=1, checkpointing=False,N_CLASSES=20):
    model = torch_models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
    msg = model.load_state_dict(torch.load(wight_path, map_location='cpu')['weights'])
    model = model.cuda()
    model = NormalizationAndCutoutWrapper(model, size=256,mean=torch.FloatTensor(mean),
                                          std=torch.FloatTensor(std), cut_power=cut_power,
                                          noise_sd=noise_sd, noise_schedule=noise_schedule,
                                          noise_descending_steps=noise_descending_steps,
                                          num_cutouts=num_cutouts, checkpointing=checkpointing, batches=batches)
    # model.eval()
    return model.cuda().eval()

def load_tv(wight_path,N_CLASSES=20):
    model = torch_models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
    msg = model.load_state_dict(torch.load(wight_path, map_location='cpu')['weights'])
    model=model.cuda()
    model = NormalizationAndResizeWrapper(model, 256,
                                          mean=torch.FloatTensor(mean),
                                          std=torch.FloatTensor(std))
    # model.eval()
    return model.cuda().eval()