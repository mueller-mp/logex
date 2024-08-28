import torch
import timm
import torchvision.transforms.functional as TF
class RandomizedSmoothingWrapper(torch.nn.Module):
    def __init__(self, model, steps, noise_sd):
        super().__init__()

        self.model = model
        self.steps = steps
        self.noise_sd = noise_sd
        self.train(model.training)

        self.model = model

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

    def forward(self, x):
        bs = x.shape[0]
        noisy_x = torch.repeat_interleave(x, self.steps, dim=0)
        noisy_x += self.noise_sd * torch.randn_like(noisy_x)
        out = self.model(noisy_x)
        out_avg = torch.zeros((bs, out.shape[1]), device=x.device)
        idx = 0
        for i in range(bs):
            out_avg[i] = torch.mean(out[idx:(idx+self.steps)], dim=0)
        return out_avg

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

