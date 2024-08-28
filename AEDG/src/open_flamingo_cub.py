import os

import argparse
import torch
from utils.datasets.bird_dataset import BirdDataset as CubDataset
import torchvision.transforms as transforms
from utils.open_flamingo import create_model_and_transforms

demo_imgs_idcs_captions = [
    (411428, 'an image of an admiral sitting on a purple flower in front of a blurry background'),
    (1278006, 'an image of bolete in the grass'),
    (1178950, 'an image of a traffic light in front of a tree and a white church'),
    (136098, 'an image of a koala hanging on a tree'),
    (1184094, 'an image of guacamole in a bowl with flatbread in the background'),
    (45134,'an image of a leatherback turtle on the beach'),
    (176902, 'an image of an European gallinule sitting in the grass'),
    (392182, 'an image of a rhinoceros beetle sitting on a tree stamp in front of a blurry background'),
    (368716, 'an image of a leopard lying on the ground in front of grass'),
    (362277, 'an image of a persian cat lying on top of a book with glasses and a chair in the background'),
    (1193941, 'an image of pretzels lying on top of a kitchen cloth'),
    (701468, 'an image of an electric guitar standing in front of wooden planks'),
    (1194988, 'an image of a cheese burger with fries next to it'),
    (257954, 'an image of a soft-coated wheaten terrier standing on the ground outside'),
    (78721, 'an image of a night snake resting on wood'),
    (373870, 'an image of a tiger jumping off wooden logs behind a fence'),
    (139796, 'an image of a sea anemone underwater on rocks'),
]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_idx', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--null_text_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--results_folder', type=str, default='output/cub_captions')
    parser.add_argument('--imagenet_folder', type=str, default='/scratch/datasets/CUB_200_2011')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    args = parser.parse_args()

    device = torch.device('cuda')
    llama_weights_hf = 'checkpoints/llama/7B-hf'
    tokenizer_weights_hf = 'checkpoints/llama/7B-hf'

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_weights_hf,
        tokenizer_path=tokenizer_weights_hf,
        cross_attn_every_n_layers=4
    )

    #checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
    checkpoint_path = 'checkpoints/OpenFlamingo-9B/checkpoint.pt'
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model = model.to(device)
    tokenizer = tokenizer

    #load demo images
    pre_crop_size = int(args.resolution * 1.25)
    transform = transforms.Compose([transforms.Resize(pre_crop_size), transforms.CenterCrop(args.resolution)])

    in_train_dataset = ImageNet(args.imagenet_folder, split='train', transform=transform)

    def get_imagenet_labels():
        classes_extended = in_train_dataset.classes
        labels = []
        for a in classes_extended:
            labels.append(a[0])
        return labels

    in_labels = get_imagenet_labels()

    num_demo_imgs = len(demo_imgs_idcs_captions)
    demo_images = torch.zeros((1, num_demo_imgs, 1, 3, args.resolution, args.resolution))

    tokenizer_demo_input = ''

    for i, (imagenet_idx, caption) in enumerate(demo_imgs_idcs_captions):
        imagenet_img, _ = in_train_dataset[imagenet_idx]
        imagenet_img = image_processor(imagenet_img)
        demo_images[0, i, 0] = imagenet_img
        tokenizer_demo_input += f'<image>{caption}.<|endofchunk|>'

    cub_val_dataset = ImageNet(args.imagenet_folder, split='val', transform=transform)


    if args.class_idx < 0 or args.class_idx > 999:
        target_classes = torch.arange(0, 1000, dtype=torch.long)
    else:
        target_classes = torch.LongTensor([args.class_idx])

    for target_classs in target_classes:
        class_label = in_labels[target_classs]
        val_class_idcs = torch.nonzero(torch.LongTensor(cub_val_dataset.targets) == target_classs, as_tuple=False).squeeze()

        # make the output folders
        class_folder = os.path.join(args.results_folder, f'{target_classs}_{class_label}')
        os.makedirs(class_folder, exist_ok=True)

        with torch.no_grad():
            for img_idx in range(args.num_images):
                in_idx = val_class_idcs[img_idx]

                caption_file = os.path.join(class_folder, f"{in_idx}_prompt.txt")
                if os.path.isfile(caption_file):
                    continue

                imagenet_img, _ = cub_val_dataset[in_idx]
                imagenet_img_processed = image_processor(imagenet_img)[None, None, None]

                vision_x = torch.cat([demo_images, imagenet_img_processed], dim=1).to(device)
                tokenizer_input = tokenizer_demo_input + f'<image>an image of a {class_label}'
                tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
                lang_x = tokenizer(
                    [tokenizer_input],
                    return_tensors="pt",
                )
                lang_x = lang_x.to(device)

                generated_text = model.generate(
                    vision_x=vision_x,
                    lang_x=lang_x["input_ids"],
                    attention_mask=lang_x["attention_mask"],
                    max_new_tokens=20,
                    num_beams=3,
                )

                generated_text = generated_text.cpu()

                text_decoded = tokenizer.decode(generated_text[0])
                image_caption = text_decoded.split('<image>')[-1]

                if '.' in image_caption:
                    image_caption = image_caption.split('.')[0]

                imagenet_img.save(os.path.join(class_folder, f"{in_idx}_original.png"))

                with open(caption_file, 'w') as f:
                    f.write(image_caption)

