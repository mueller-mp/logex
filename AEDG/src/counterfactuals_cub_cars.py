import torch

from diffusers import DDIMScheduler

import os
import argparse
import torchvision.transforms as transforms
from PIL import Image

from utils.models.load_mae_model import load_cub_mae, load_cub_mae_with_cutout
from utils.models.load_cal_model import load_cal_model, load_cal_model_with_cutout
from utils.sd_backprop_pipe import StableDiffusionPipelineWithGrad
from utils.plotting_utils import plot, plot_attention
from utils.loss_functions import get_loss_function, calculate_confs
from utils.cub_classes import IDX2NAME as cub_labels
from utils.cars_classes import IDX2NAME as cars_labels

from utils.datasets.car_dataset import CarDataset
from utils.datasets.bird_dataset import BirdDataset

cub_vces_img_id_start_target = [
    # (1271,46,66),
    # (1272,46,66),
    # (1273,46,66),
    # (1274,46,66),
    # (1276,46,66),
    #
    # (1271,46,67),
    # (1272,46,67),
    # (1273,46,67),
    # (1274,46,67),
    # (1276,46,67),
    #
    # (1271,46,72),
    # (1272,46,72),
    # (1273,46,72),
    # (1274,46,72),
    # (1276,46,72),
    #
    # (1271,46,117),
    # (1272,46,117),
    # (1273,46,117),
    # (1274,46,117),
    # (1276,46,117),

    (391, 15, 150),
    (393, 15, 150),
    (399, 15, 150),
    (401, 15, 150),
    (401, 15, 150),

    (391, 15, 151),
    (393, 15, 151),
    (399, 15, 151),
    (401, 15, 151),
    (401, 15, 151),

    (391, 15, 152),
    (393, 15, 152),
    (399, 15, 152),
    (401, 15, 152),
    (401, 15, 152),

    (391, 15, 153),
    (393, 15, 153),
    (399, 15, 153),
    (401, 15, 153),
    (401, 15, 153),

    (391, 15, 158),
    (393, 15, 158),
    (399, 15, 158),
    (401, 15, 158),
    (401, 15, 158),

    (391, 15, 160),
    (393, 15, 160),
    (399, 15, 160),
    (401, 15, 160),
    (401, 15, 160),

    (391, 15, 163),
    (393, 15, 163),
    (399, 15, 163),
    (401, 15, 163),
    (401, 15, 163),

    (391, 15, 169),
    (393, 15, 169),
    (399, 15, 169),
    (401, 15, 169),
    (401, 15, 169),
]

car_vces_img_id_start_target = [
    (929,55,143),
    (1741,55,143),

    (929,55,172),
    (1741,55,172),

    (929,55,111),
    (1741,55,111),

    (929,55,33),
    (1741,55,33),

    (929,55,34),
    (1741,55,34),

    (929,55,44),
    (1741,55,44),

    (929,55,14),
    (1741,55,14),
]


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--local_split', type=int, default=0)
    parser.add_argument('--num_splits', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='cub')
    parser.add_argument('--results_folder', type=str, default='output/')
    parser.add_argument('--inversion_folder', type=str, default=None)
    parser.add_argument('--dataset_folder', type=str, default='/scratch/datasets/CUB_200_2011')

    parser.add_argument('--regularizers', nargs='+', default=[], type=str)
    parser.add_argument('--regularizers_ws', nargs='+', default=[], type=float)
    parser.add_argument('--p2p', action='store_true')
    parser.add_argument('--segmentation_squash', type=str, default=None)

    #parser.add_argument('--solver', type=str, default='ddim')
    parser.add_argument('--loss', type=str, default='log_conf')
    parser.add_argument('--sgd_steps', type=int, default=20)
    parser.add_argument('--sgd_stepsize', type=float, default=0.05)
    parser.add_argument('--sgd_schedule', type=str, default='cosine')
    parser.add_argument('--normalize_gradient', action='store_true')
    parser.add_argument('--guidance_scale', type=float, default=3.0)
    parser.add_argument('--early_stopping_confidence', type=float, default=None)
    parser.add_argument('--conditioning_lr_factor', type=float, default=1.0)
    parser.add_argument('--uncond_lr_factor', type=float, default=1.0)
    parser.add_argument('--latent_lr_factor', type=float, default=1.0)
    parser.add_argument('--num_ddim_steps', type=int, default=20)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--use_double', action='store_true')
    parser.add_argument('--optimize_latents', action='store_true')
    parser.add_argument('--optimize_conditioning', action='store_true')
    parser.add_argument('--optimize_uncond', action='store_true')
    parser.add_argument('--per_timestep_conditioning', action='store_true')
    parser.add_argument('--per_timestep_uncond', action='store_true')
    parser.add_argument('--conditioning_gradient_masking', action='store_true')
    parser.add_argument('--xa_interpolation_schedule', type=str, default=None)
    parser.add_argument('--self_interpolation_schedule', type=str, default=None)
    parser.add_argument('--augmentation_num_cutouts', type=int, default=32)
    parser.add_argument('--augmentation_noise_sd', type=float, default=0)
    parser.add_argument('--augmentation_noise_schedule', type=str, default='const')
    parser.add_argument('--random_images', action='store_true')

    args = parser.parse_args()

    #torch.set_deterministic_debug_mode(1)
    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(args.seed)

    num_cutouts = args.augmentation_num_cutouts

    dataset = args.dataset
    if dataset == 'cub':
        class_labels = cub_labels
        num_classes = len(class_labels)
        class_sort_idcs = torch.arange(0, num_classes, dtype=torch.long)
        if num_cutouts > 0:
            augm_string = f'_cutout_{num_cutouts}'
            classifier = load_cub_mae_with_cutout(0.3, num_cutouts, checkpointing=True,
                                                  noise_sd=args.augmentation_noise_sd,
                                                  noise_schedule=args.augmentation_noise_schedule,
                                                  noise_descending_steps=args.sgd_steps)
        else:
            augm_string = '_no_aug'
            classifier = load_cub_mae()

        classifier.to(device)

        parent_result_folder = os.path.join(args.results_folder, 'cub_counterfactuals')
        img_id_start_target = cub_vces_img_id_start_target
    elif dataset == 'cars':
        class_labels = cars_labels
        num_classes = len(class_labels)
        if num_cutouts > 0:
            augm_string = f'_cutout_{num_cutouts}'
            classifier = load_cal_model_with_cutout('cars', 0.3, num_cutouts, checkpointing=True,
                                                    noise_sd=args.augmentation_noise_sd,
                                                    noise_schedule=args.augmentation_noise_schedule,
                                                    noise_descending_steps=args.sgd_steps)
        else:
            augm_string = '_no_aug'
            classifier = load_cal_model('cars')

        parent_result_folder = os.path.join(args.results_folder, 'cars_counterfactuals')
        img_id_start_target = car_vces_img_id_start_target
    else:
        raise NotImplementedError()

    if args.random_images:
        if args.dataset == 'cub':
            dataset = BirdDataset(args.dataset_folder, phase='test', transform=None)
        elif args.dataset == 'cars':
            dataset = CarDataset(args.dataset_folder, phase='test', transform=None)
        else:
            raise NotImplementedError()

        dataset_targets = dataset.targets
        img_id_start_target = []
        torch.manual_seed(args.gpu)
        for i in range(args.num_images):
            img_idx = torch.randint(0, len(dataset_targets), (1,), dtype=torch.long).item()
            start_class = dataset_targets[img_idx]
            target_class = torch.randint(0, num_classes, (1,)).item()
            img_id_start_target.append((img_idx, start_class, target_class))

    classifier.to(device)

    pipe = StableDiffusionPipelineWithGrad.from_pretrained(args.model_path)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if args.use_double:
        pipe.convert_to_double()

    pipe.to(device)

    loss_function = get_loss_function('log_conf', classifier)
    to_tensor = transforms.ToTensor()

    assert len(args.regularizers) == len(args.regularizers_ws)
    regularizers_weights = {}
    reg_string = ''
    for reg_name, reg_w in zip(args.regularizers, args.regularizers_ws):
        regularizers_weights[reg_name] = reg_w
        reg_string += f'_{reg_name}_{reg_w}'

    p2p = args.p2p
    xa_interpolation_schedule = args.xa_interpolation_schedule
    self_interpolation_schedule = args.self_interpolation_schedule
    sgd_steps = args.sgd_steps
    sgd_stepsize = args.sgd_stepsize
    sgd_schedule = args.sgd_schedule
    optimize_latents = args.optimize_latents
    optimize_conditioning = args.optimize_conditioning
    optimize_uncond = args.optimize_uncond
    per_timestep_conditioning_delta = args.per_timestep_conditioning
    per_timestep_uncond_delta = args.per_timestep_uncond
    latent_lr_factor = args.latent_lr_factor
    conditioning_lr_factor = args.conditioning_lr_factor
    uncond_lr_factor = args.uncond_lr_factor
    conditioning_gradient_masking = 'foreground' if args.conditioning_gradient_masking else None
    segmentation_squash = args.segmentation_squash
    loss = args.loss
    output_description = f'ddim_{args.num_ddim_steps}_{loss}_sgd_{sgd_steps}_{sgd_stepsize}_{sgd_schedule}'
    output_description += f'_squash_{segmentation_squash}' if segmentation_squash is not None else ''
    if p2p:
        output_description += '_p2p'
    if args.normalize_gradient:
        output_description += '_normalize'
    if optimize_latents:
        output_description += f'_latents'
        output_description += f'_{latent_lr_factor}' if latent_lr_factor != 1.0 else ''
    if optimize_conditioning:
        output_description += f'_cond'
        output_description += '_pt' if per_timestep_conditioning_delta else ''
        output_description += f'_{conditioning_lr_factor}' if conditioning_lr_factor != 1.0 else ''
        output_description += '_conditioning_gradient_masking' if conditioning_gradient_masking else ''
    if optimize_uncond:
        output_description += f'_uncond'
        output_description += '_pt' if per_timestep_uncond_delta else ''
        output_description += f'_{uncond_lr_factor}' if uncond_lr_factor != 1.0 else ''
    if xa_interpolation_schedule:
        output_description += f'_xa_interpolation_{xa_interpolation_schedule}'
    if xa_interpolation_schedule:
        output_description += f'_self_interpolation_{self_interpolation_schedule}'
    output_description += reg_string
    output_description += augm_string

    to_pil = transforms.ToPILImage()

    for linear_idx, (img_idx, start_class, target_class) in enumerate(img_id_start_target):
        if args.num_splits > 1:
            current_split_idx = linear_idx % args.num_splits
            if not current_split_idx == args.local_split:
                continue

        start_label = class_labels[start_class]
        target_label = class_labels[target_class]

        if p2p:
            prompt_to_prompt_replacements = (start_label, target_label)
        else:
            prompt_to_prompt_replacements = None
        start_class_folder = os.path.join(args.inversion_folder, f'{start_class}_{start_label}')
        inversion_dir = os.path.join(start_class_folder,
                                     f"inversion_{args.num_ddim_steps}_{args.guidance_scale}")

        result_folder = os.path.join(args.results_folder, output_description,
                                     f'{start_class}_{start_label}/')
        os.makedirs(result_folder, exist_ok=True)

        out_pdf = os.path.join(result_folder, f'{img_idx}_{target_class}_{target_label}.pdf')
        if os.path.isfile(out_pdf):
            continue

        try:
            latent = torch.load(os.path.join(inversion_dir, f'{img_idx}.pt'), map_location='cpu')[None, :]
            null_texts_embeddings = torch.load(os.path.join(inversion_dir, f'{img_idx}_null_texts.pt'), map_location='cpu')
            assert len(null_texts_embeddings) == args.num_ddim_steps

            if args.use_double:
                latent = latent.double()
                null_texts_embeddings = null_texts_embeddings.double()

            captions_file = os.path.join(inversion_dir, f'{img_idx}_prompt.txt')
            if os.path.isfile(captions_file):
                with open(captions_file, 'r') as f:
                    prompt_str = f.read()
            else:
                print(f'Warning: Could not load caption from {captions_file}')
                continue


            prompt_foreground_key_words = [word for word in start_label.split()]
            if True:
                prompt_to_prompt_str = prompt_str.replace(start_label, target_label)

            original_img = Image.open(os.path.join(inversion_dir, f'{img_idx}_original.png'))
            original_tensor = to_tensor(original_img).to(device)
            if args.use_double:
                original_tensor = original_tensor.double()

            reconstructed_img = Image.open(os.path.join(inversion_dir, f'{img_idx}_null_reconstructed.png'))
            reconstructed_tensor = to_tensor(reconstructed_img)

            return_values = pipe(target_class,
                                 prompt=prompt_str,
                                 prompt_foreground_key_words=prompt_foreground_key_words,
                                 prompt_to_prompt_replacements=prompt_to_prompt_replacements,
                                 squash_function=segmentation_squash,
                                 starting_img=original_tensor,
                                 latents=latent,
                                 regularizers_weights=regularizers_weights,
                                 sgd_steps=sgd_steps,
                                 sgd_stepsize=sgd_stepsize,
                                 latent_lr_factor=latent_lr_factor,
                                 conditioning_lr_factor=conditioning_lr_factor,
                                 uncond_lr_factor=uncond_lr_factor,
                                 cond_gradient_masking=conditioning_gradient_masking,
                                 xa_interpolation_schedule=xa_interpolation_schedule,
                                 self_interpolation_schedule=self_interpolation_schedule,
                                 early_stopping_loss=None,
                                 sgd_optim='adam',
                                 sgd_scheduler=sgd_schedule,
                                 normalize_gradients=args.normalize_gradient,
                                 gradient_clipping=None,
                                 loss_function=loss_function,
                                 guidance_scale=args.guidance_scale,
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
            px_foreground_segmentation = return_values['px_foreground_segmentation']
            words_attention_masks = return_values['words_attention_masks']

            start_confs = calculate_confs(classifier, img_grid, device, target_class=start_class)
            target_confs = calculate_confs(classifier, img_grid, device, target_class=target_class)

            confs = torch.cat([start_confs[:, None], target_confs[:, None]], dim=1)

            min_loss_idx = torch.argmin(torch.FloatTensor(loss_scores)).item()

            min_loss_img = img_grid[min_loss_idx]
            #torch.save(min_loss_img, os.path.join(result_folder, f'{img_idx}.pth'))

            original_img.save(os.path.join(result_folder, f'{img_idx}.png'))
            pil_sd = to_pil(torch.clamp(img_grid[0], 0, 1))
            pil_sd.save(os.path.join(result_folder, f'{img_idx}_{target_class}_{target_label}_sd.png'))
            pil_ours = to_pil(torch.clamp(min_loss_img, 0, 1))
            pil_ours.save(os.path.join(result_folder, f'{img_idx}_{target_class}_{target_label}_ours.png'))

            title_attribute = ''
            max_class_name_length = 8
            if len(start_label) > max_class_name_length:
                title_attribute += start_label[:max_class_name_length] + '.'
            else:
                title_attribute += start_label
            title_attribute += ' - '
            if len(target_label) > max_class_name_length:
                title_attribute += target_label[:max_class_name_length] + '.'
            else:
                title_attribute += target_label

            plot(original_tensor, reconstructed_tensor, img_grid, title_attribute, confs, out_pdf,
                 loss_scores=loss_scores, regularizer_scores=regularizer_scores)
            plot_attention(original_tensor, px_foreground_segmentation, words_attention_masks,
                           os.path.join(result_folder, f'{img_idx}_attention.pdf'))
        except:
            continue