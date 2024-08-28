import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint as checkpoint
from torchvision.transforms import functional as TF

# IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

IMAGENET_DEFAULT_MEAN = [0.5, 0.5, 0.5]
IMAGENET_DEFAULT_STD = [0.5, 0.5, 0.5]

class NormalizationAndCutoutWrapper(torch.nn.Module):
    def __init__(self, model, size, mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0]),
                 cut_power=1.0, num_cutouts=16, checkpointing=False):
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
        self.checkpointing = checkpointing
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

    def forward(self, x, *args, augment=True):
        def chkpt_function(x, *args):
            if augment:
                x = self.cutout(x, self.num_cutouts)
            else:
                x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BICUBIC)
            return self.model((x - self.mean) / self.std, *args)

        if self.checkpointing:
            out = checkpoint.checkpoint(chkpt_function, x, *args)
        else:
            out = chkpt_function(x, *args)
        return out


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

    def forward(self, x, *args, augment=False):
        if self.size is not None:
            x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BICUBIC)

        return self.model((x - self.mean) / self.std, *args)


    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)


class NormalizationAndRandomNoiseWrapper(torch.nn.Module):
    def __init__(self, model, size, num_samples, noise_sd, mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0]), checkpointing=False):
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
        self.checkpointing = checkpointing

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

    def forward(self, x, *args, augment=True):
        if self.size is not None:
            x = TF.resize(x, self.size, interpolation=TF.InterpolationMode.BICUBIC)

        if augment:
            noisy_x = x.expand(self.num_samples, -1, -1, -1)
            noisy_x = noisy_x + self.noise_sd * torch.randn_like(noisy_x)
        else:
            noisy_x = x

        out = self.model(noisy_x, *args)
        return out

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)
