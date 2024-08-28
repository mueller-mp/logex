from .metaformer_models.config import get_config
from .metaformer_models import build_model
import os
import torch
import importlib
import torch.distributed as dist
from .model_wrappers import NormalizationAndCutoutWrapper, NormalizationAndResizeWrapper, \
    NormalizationAndRandomNoiseWrapper, IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    print(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    if 'model' not in checkpoint:
        if 'state_dict_ema' in checkpoint:
            checkpoint['model'] = checkpoint['state_dict_ema']
        else:
            checkpoint['model'] = checkpoint
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        print(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

class FakeArgs():
    def __init__(self, dataset, resume, cfg):
        self.opts = False
        self.batch_size = False
        self.data_path = False
        self.zip = False
        self.cache_mode = False
        self.accumulation_steps = False
        self.use_checkpoint = False
        self.amp_opt_level = False
        self.output = False
        self.tag = False
        self.eval = True
        self.throughput = False

        self.num_workers = False

        # set lr and weight decay
        self.lr = False
        self.min_lr = False
        self.warmup_lr = False
        self.warmup_epochs = False
        self.weight_decay = False

        self.epochs = False
        self.lr_scheduler_name = False
        self.pretrain = False
        self.local_rank = False

        self.dataset = dataset
        self.resume = resume
        self.cfg = cfg

def load_inat2021_metaformer(device, size=0):
    if size == 0:
        cfg = 'metaformer_ckpts/configs/MetaFG_0_inat2021.yaml'
        chkpt = 'metaformer_ckpts/metafg_0_inat21_384.pth'
    elif size == 1:
        cfg = 'metaformer_ckpts/configs/MetaFG_1_inat2021.yaml'
        chkpt = 'metaformer_ckpts/metafg_1_inat21_384.pth'
    else:
        raise NotImplementedError()

    args = FakeArgs('inaturelist2021', chkpt, cfg)

    config = get_config(args)
    model = build_model(config)
    max_accuracy = load_checkpoint(config, model, None, None, None)
    model.add_meta = False
    model = model.eval()
    model = NormalizationAndResizeWrapper(model, 384,
                                          mean=torch.FloatTensor(IMAGENET_DEFAULT_MEAN),
                                          std=torch.FloatTensor(IMAGENET_DEFAULT_STD))
    model.to(device)
    out = model(torch.randn((2, 3, 384, 384), device=device))
    return model

def load_inat2021_metaformer_with_cutout(device, cut_power, num_cutouts, size=0, checkpointing=False):
    if size == 0:
        cfg = 'metaformer_ckpts/configs/MetaFG_0_inat2021.yaml'
        chkpt = 'metaformer_ckpts/metafg_0_inat21_384.pth'
    else:
        raise NotImplementedError()

    args = FakeArgs('inaturelist2021', chkpt, cfg)

    config = get_config(args)
    model = build_model(config)
    max_accuracy = load_checkpoint(config, model, None, None, None)
    model = model.eval()
    model.to(device)
    model = NormalizationAndCutoutWrapper(model, 384,
                                          mean=torch.FloatTensor(IMAGENET_DEFAULT_MEAN),
                                          std=torch.FloatTensor(IMAGENET_DEFAULT_STD),
                                          cut_power=cut_power, num_cutouts=num_cutouts,
                                          checkpointing=checkpointing)
    return model