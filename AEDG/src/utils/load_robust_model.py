import torch
import torchvision.models as torch_models
import os
class NormalizationWrapper(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()

        mean = torch.tensor(mean)
        std = torch.tensor(std)

        mean = mean[..., None, None]
        std = std[..., None, None]

        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x, *args, **kwargs):
        x_normalized = (x - self.mean)/self.std
        return self.model(x_normalized, *args, **kwargs)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

def ImageNetWrapper(model):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return NormalizationWrapper(model, mean, std)

def load_madry_l2(modelname, epoch, device):
    import torch
    models_dir = '/mnt/SHARED/Max/RobustFinetuning/trained_models'
    model = torch_models.resnet50()
    state_dict_file = os.path.join(models_dir, modelname, f'ep_{epoch}.pth')
    state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    last_layer = model.fc

    model = ImageNetWrapper(model)
    model.eval()
    model = model.to(device)

    return model, last_layer
