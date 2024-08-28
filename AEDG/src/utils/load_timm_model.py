import torch
import timm
import torchvision.transforms.functional as TF
from .make_cutout import MakeCutouts

class NormalizationAndCutoutWrapper(torch.nn.Module):
    def __init__(self, model, size, mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0]),
                 cut_power=1.0, num_cutouts=16):
        super().__init__()

        self.model = model

        mean = mean[..., None, None]
        std = std[..., None, None]

        self.size = size
        self.cutout = MakeCutouts(size, cut_power)
        self.num_cutouts = num_cutouts
        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @property
    def return_layers(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self.model.model.model._return_layers

    @return_layers.setter
    def return_layers(self, value):
        print("setter of x called")
        self.model.model.model._return_layers = value

    @return_layers.deleter
    def return_layers(self):
        print("deleter of x called")
        del self.model.model.model._return_layers

    def forward(self, x, augment=True):
        if augment:
            x = self.cutout(x, self.num_cutouts)
        else:
            x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BILINEAR)

        return self.model(
            (x - self.mean) / self.std
        )

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

class NormalizationAndResizeWrapper(torch.nn.Module):
    def __init__(self, model, size, mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])):
        super().__init__()

        self.model = model

        mean = mean[..., None, None]
        std = std[..., None, None]

        self.size = size
        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @property
    def return_layers(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self.model.model.model._return_layers

    @return_layers.setter
    def return_layers(self, value):
        print("setter of x called")
        self.model.model.model._return_layers = value

    @return_layers.deleter
    def return_layers(self):
        print("deleter of x called")
        del self.model.model.model._return_layers

    def forward(self, x, augment=False):
        if self.size is not None:
            x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BILINEAR)

        return self.model(
            (x - self.mean) / self.std
        )

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

class NormalizationAndRandomNoiseWrapper(torch.nn.Module):
    def __init__(self, model, size, num_samples, noise_sd, mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])):
        super().__init__()

        self.model = model
        self.num_samples = num_samples
        self.noise_sd = noise_sd
        self.train(model.training)

        self.size = size
        self.model = model

        mean = mean[..., None, None]
        std = std[..., None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @property
    def return_layers(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self.model.model.model._return_layers

    @return_layers.setter
    def return_layers(self, value):
        print("setter of x called")
        self.model.model.model._return_layers = value

    @return_layers.deleter
    def return_layers(self):
        print("deleter of x called")
        del self.model.model.model._return_layers

    def forward(self, x, augment=True):
        if self.size is not None:
            x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BILINEAR)

        if augment:
            noisy_x = x.expand(self.num_samples, -1, -1, -1)
            noisy_x = noisy_x + self.noise_sd * torch.randn_like(noisy_x)
        else:
            noisy_x = x

        out = self.model(noisy_x)
        return out

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

def load_timm_model_with_cutout(model_name, cut_power, num_cutouts):
    model = timm.create_model(model_name, pretrained=True)
    cfg = model.default_cfg
    assert cfg["input_size"][1] == cfg["input_size"][2]
    size = cfg["input_size"][1]
    model = NormalizationAndCutoutWrapper(model, size=size, mean=torch.tensor(cfg["mean"]),
                                          std=torch.tensor(cfg["std"]), cut_power=cut_power, num_cutouts=num_cutouts)
    return model
def load_timm_model_with_random_noise(model_name, noise_sd, num_samples):
    model = timm.create_model(model_name, pretrained=True)
    cfg = model.default_cfg
    assert cfg["input_size"][1] == cfg["input_size"][2]
    size = cfg["input_size"][1]
    model = NormalizationAndRandomNoiseWrapper(model, size=size, mean=torch.tensor(cfg["mean"]),
                                               std=torch.tensor(cfg["std"]), noise_sd=noise_sd, num_samples=num_samples)
    return model
def load_timm_model(model_name, auto_resize=True):
    model = timm.create_model(model_name, pretrained=True)
    cfg = model.default_cfg
    assert cfg["input_size"][1] == cfg["input_size"][2]
    if auto_resize:
        size = cfg["input_size"][1]
    else:
        size = None
    model = NormalizationAndResizeWrapper(model, size=size, mean=torch.tensor(cfg["mean"]),
                                          std=torch.tensor(cfg["std"]))

    return model