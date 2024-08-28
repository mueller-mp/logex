import os, pdb

import argparse
import numpy as np
import torch
import requests
from PIL import Image

#from utils.scheduler import DDIMInverseScheduler
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from utils.null_text_inversion import NullInversion
from utils.datasets.bird_dataset import BirdDataset as CubDataset
from utils.datasets.car_dataset import CarDataset as CarsDataset
from utils.cub_classes import IDX2NAME as cub_labels
from utils.cars_classes import IDX2NAME as cars_labels

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cub')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--null_text_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--results_folder', type=str, default=None)
    parser.add_argument('--flamingo_captions_folder', type=str, default=None)
    parser.add_argument('--dataset_folder', type=str, default='/scratch/datasets/CUB_200_2011')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--local_split', type=int, default=0)
    parser.add_argument('--num_splits', type=int, default=0)
    args = parser.parse_args()

    assert args.results_folder is not None
    assert args.flamingo_captions_folder is not None

    pre_crop_size = int(args.resolution * 1.25)
    transform = transforms.Compose([transforms.Resize(pre_crop_size), transforms.CenterCrop(args.resolution)])

    if args.dataset == 'cub':
        in_dataset = CubDataset(args.dataset_folder, phase='test', transform=transform)
        in_labels = cub_labels
    elif args.dataset == 'cars':
        in_dataset = CarsDataset(args.dataset_folder, phase='test', transform=transform)
        in_labels = cars_labels
    else:
        raise NotImplementedError()

    device = torch.device(f'cuda:{args.gpu}')
    forward_pipe = StableDiffusionPipeline.from_pretrained(args.model_path).to(device)
    forward_pipe.scheduler = DDIMScheduler.from_config(forward_pipe.scheduler.config)
    forward_pipe.safety_checker = lambda images, **kwargs: (images, False)

    null_inversion = NullInversion(forward_pipe, args.num_ddim_steps, args.guidance_scale)


    to_do_list_class_img = []
    dataset_targets = torch.LongTensor(in_dataset.targets)
    dataset_targets_sord_idcs = torch.argsort(dataset_targets, descending=False)
    for in_idx in dataset_targets_sord_idcs:
        target_class = dataset_targets[in_idx]
        to_do_list_class_img.append((target_class.item(), in_idx.item()))

    print(f'Inverting {len(to_do_list_class_img)} images')

    for linear_idx, (target_class, in_idx) in enumerate(to_do_list_class_img):
        if args.num_splits > 1:
            current_split_idx = linear_idx % args.num_splits
            if not current_split_idx == args.local_split:
                continue

            class_label = in_labels[target_class]
            class_folder = os.path.join(args.results_folder, f'{target_class}_{class_label}')

            # make the output folders
            inversion_folder = os.path.join(class_folder, f"inversion_{args.num_ddim_steps}_{args.guidance_scale}")
            os.makedirs(inversion_folder, exist_ok=True)
            flamingo_class_captions_folder = os.path.join(args.flamingo_captions_folder, f'{target_class}_{class_label}')

            # if the input is a folder, collect all the images as a list
            if os.path.isfile(os.path.join(inversion_folder, f"{in_idx}_original.png")):
                continue
            try:
                img, _ = in_dataset[in_idx]
                img_np = np.array(img)

                captions_file = os.path.join(flamingo_class_captions_folder, f'{in_idx}_prompt.txt')
                if os.path.isfile(captions_file):
                    with open(captions_file, 'r') as f:
                        prompt_str = f.read()
                else:
                    print(f'Warning: Could not load caption from {captions_file}')
                    continue

                (_, x_null_reconstructed), x_inv, null_texts = null_inversion.invert(img_np, prompt_str)
                x_null_reconstructed = Image.fromarray(x_null_reconstructed)
                x_reconstructed = forward_pipe(prompt_str, height=args.resolution, width=args.resolution,
                                               guidance_scale=1, num_inference_steps=args.num_ddim_steps, latents=x_inv)
                x_reconstructed = x_reconstructed.images

                # save the inversion
                img.save(os.path.join(inversion_folder, f"{in_idx}_original.png"))
                torch.save(torch.stack(null_texts), os.path.join(inversion_folder, f"{in_idx}_null_texts.pt"))
                torch.save(x_inv[0], os.path.join(inversion_folder, f"{in_idx}.pt"))
                x_null_reconstructed.save(os.path.join(inversion_folder, f"{in_idx}_null_reconstructed.png"))
                x_reconstructed[0].save(os.path.join(inversion_folder, f"{in_idx}_reconstructed.png"))
                # save the prompt string
                with open(os.path.join(inversion_folder, f"{in_idx}_prompt.txt"), "w") as f:
                    f.write(prompt_str)
            except:
                print(f'Could not invert image {in_idx} - {class_label}  ({target_class}) - skipping')
