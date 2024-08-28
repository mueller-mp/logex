import math
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import wandb


def plot(original_image, original_reconstructed, trajectory_imgs, title_attribute, attribute_values_over_trajectory,
         filename, loss_scores=None, regularizer_scores=None, model_grad_cams=None, wandb_name=None, wandb_step=None, preds=None, confs=None):
    scale_factor = 4.0
    num_cols = max(2, math.ceil(math.sqrt(len(trajectory_imgs))))
    num_rows = math.ceil(len(trajectory_imgs) / num_cols)

    if model_grad_cams is not None:
        num_sub_rows = 1 + model_grad_cams.shape[0]
    else:
        num_sub_rows = 1

    total_rows = 1 + num_sub_rows * num_rows

    fig, axs = plt.subplots(total_rows, num_cols, figsize=(scale_factor * num_cols, total_rows * 1.3 * scale_factor))

    # plot original:
    axs[0, 0].axis('off')
    if original_image is not None:
        img = original_image.permute(1, 2, 0).cpu().detach()
        axs[0, 0].set_title('Original')
        axs[0, 0].imshow(img, interpolation='lanczos')

    axs[0, 1].axis('off')
    if original_reconstructed is not None:
        img = original_reconstructed.permute(1, 2, 0).cpu().detach()
        axs[0, 1].set_title('Original Null-Reconstructed')
        axs[0, 1].imshow(img, interpolation='lanczos')

    for j in range(2, num_cols):
        axs[0, j].axis('off')

    # plot counterfactuals
    for outer_row_idx in range(0, num_rows):
        row_idx = 1 + outer_row_idx * num_sub_rows
        for sub_row_idx in range(num_sub_rows):
            for col_idx in range(num_cols):
                img_idx = outer_row_idx * num_cols + col_idx
                ax = axs[row_idx + sub_row_idx, col_idx]
                if img_idx >= len(trajectory_imgs):
                    ax.axis('off')
                    continue
                if sub_row_idx == 0:
                    img = trajectory_imgs[img_idx]
                    img = torch.clamp(img.permute(1, 2, 0), min=0.0, max=1.0)

                    ax.axis('off')
                    ax.imshow(img, interpolation='lanczos')

                    title = ''
                    if title_attribute is not None:
                        vals = attribute_values_over_trajectory[img_idx]
                        if isinstance(title_attribute, list):
                            title_attribute_img = title_attribute[img_idx]
                        else:
                            title_attribute_img = title_attribute

                        title += f'{img_idx} - {title_attribute_img}'
                        if vals.dim() == 0:
                            title += f': {vals: .3f}'
                        else:
                            for val in vals:
                                title += f': {val: .3f}'
                    if loss_scores is not None:
                        title += f'\nloss: {loss_scores[img_idx]:.5f}'
                    if regularizer_scores is not None:
                        for reg_name, reg_s in regularizer_scores.items():
                            title += f'\n{reg_name}: {reg_s[img_idx]:.5f}'
                    if preds is not None and confs is not None:
                        print(preds[img_idx])
                        print(confs[img_idx])
                        title+=f'\n{preds[img_idx]}: {confs[img_idx]:.3f}'
                    ax.set_title(title)
                else:
                    #heatmap
                    img = trajectory_imgs[img_idx]
                    img = torch.clamp(img.permute(1, 2, 0), min=0.0, max=1.0)
                    cam = model_grad_cams[sub_row_idx - 1, img_idx]
                    #chw to hwc
                    img_np = img.numpy()
                    cam_np = cam.permute(1, 2, 0).numpy()

                    colormap = cv2.COLORMAP_JET
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), colormap)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    heatmap = np.float32(heatmap) / 255

                    image_weight = 0.5
                    cam = (1 - image_weight) * heatmap + image_weight * img_np
                    cam = cam / np.max(cam)

                    ax.axis('off')
                    ax.set_title(f'CAM classifier {sub_row_idx - 1}')

                    ax.imshow(cam, interpolation='lanczos')

    plt.tight_layout()
    fig.savefig(filename)

    if wandb_name is not None:
        wandb.log({wandb_name: fig}, step=wandb_step)

    plt.close(fig)

def plot_maha(original_image, original_reconstructed, trajectory_imgs, title_attribute, attribute_values_over_trajectory,
         filename, loss_scores=None, regularizer_scores=None, model_grad_cams=None, wandb_name=None, wandb_step=None, attribute_values_over_trajectory_2 = None, preds = None, preds_maha=None, confs=None, t1_maha=None, t2_maha=None, t1_msp=None, t2_msp=None, loss_scores_raw=None):
    scale_factor = 4.0
    num_cols = max(2, math.ceil(math.sqrt(len(trajectory_imgs))))
    num_rows = math.ceil(len(trajectory_imgs) / num_cols)

    if model_grad_cams is not None:
        num_sub_rows = 1 + model_grad_cams.shape[0]
    else:
        num_sub_rows = 1

    total_rows = 1 + num_sub_rows * num_rows

    fig, axs = plt.subplots(total_rows, num_cols, figsize=(scale_factor * num_cols, total_rows * 1.3 * scale_factor))

    # plot original:
    axs[0, 0].axis('off')
    if original_image is not None:
        img = original_image.permute(1, 2, 0).cpu().detach()
        axs[0, 0].set_title('Original')
        axs[0, 0].imshow(img, interpolation='lanczos')

    axs[0, 1].axis('off')
    if original_reconstructed is not None:
        img = original_reconstructed.permute(1, 2, 0).cpu().detach()
        axs[0, 1].set_title('Original Null-Reconstructed')
        axs[0, 1].imshow(img, interpolation='lanczos')

    for j in range(2, num_cols):
        axs[0, j].axis('off')

    # plot counterfactuals
    for outer_row_idx in range(0, num_rows):
        row_idx = 1 + outer_row_idx * num_sub_rows
        for sub_row_idx in range(num_sub_rows):
            for col_idx in range(num_cols):
                img_idx = outer_row_idx * num_cols + col_idx
                ax = axs[row_idx + sub_row_idx, col_idx]
                if img_idx >= len(trajectory_imgs):
                    ax.axis('off')
                    continue
                if sub_row_idx == 0:
                    img = trajectory_imgs[img_idx]
                    img = torch.clamp(img.permute(1, 2, 0), min=0.0, max=1.0)

                    ax.axis('off')
                    ax.imshow(img, interpolation='lanczos')

                    title = ''
                    if title_attribute is not None:
                        vals = attribute_values_over_trajectory[img_idx]
                        if isinstance(title_attribute, list):
                            title_attribute_img = title_attribute[img_idx]
                        else:
                            title_attribute_img = title_attribute

                        title += f'{img_idx} - {title_attribute_img} \n'
                        if vals.dim() == 0:
                            title += f': {vals: .3f}'
                        else:
                            oods = torch.full(vals.shape,float('nan'))
                            if t1_maha is not None:
                                oods[0] = vals[0]<t1_maha
                                title += f"M1 Maha: {vals[0]: .1f} {'OOD' if vals[0]<t1_maha else 'ID'} {preds_maha[img_idx,0]}"
                            if t2_maha is not None:
                                oods[1] = vals[1]<t2_maha
                                title += f" \n M2 Maha: {vals[1]: .1f} {'OOD' if vals[1]<t2_maha else 'ID'} {preds_maha[img_idx,1]}"
                            if len(vals)>=2:
                                title+='\n '
                                for val in vals[2:]:
                                    title += f' {val: .3f}' 
                            # for idx,(val,ood) in enumerate(zip(vals,oods)):
                            #     title += f' {val: .3f} '+('' if torch.isnan(ood) else ('OOD' if ood else 'ID'))

                    if attribute_values_over_trajectory_2 is not None:
                        vals = attribute_values_over_trajectory_2[img_idx]
                        low_title='w/o aug and clamp: \n '
                        if vals.dim() == 0:
                            low_title += f' {vals: .3f}'
                        else:
                            for val in vals:
                                low_title += f': {val: .3f}'
                        # ax.set_title(low_title,y=-0.11)
                    if preds is not None and confs is not None and t1_msp is not None and t2_msp is not None:
                        title+='\n' + ' \n '.join([f"M{str(idx+1)} MSP {conf:.3f} {'OOD' if conf<t else 'ID'} {str(pred)}" for idx, (pred, t, conf) in enumerate(zip(preds[img_idx,:],[t1_msp,t2_msp],confs[img_idx,:]))])
                    if loss_scores is not None:
                        title += f'\nloss: {loss_scores[img_idx]:.1f}'
                    if loss_scores_raw is not None:
                        title+=f' {loss_scores_raw[img_idx]:.1f}'
                    if regularizer_scores is not None:
                        for reg_name, reg_s in regularizer_scores.items():
                            title += f'\n{reg_name}: {reg_s[img_idx]:.5f}'
                    ax.set_title(title,loc='left')
                else:
                    #heatmap
                    img = trajectory_imgs[img_idx]
                    img = torch.clamp(img.permute(1, 2, 0), min=0.0, max=1.0)
                    cam = model_grad_cams[sub_row_idx - 1, img_idx]
                    #chw to hwc
                    img_np = img.numpy()
                    cam_np = cam.permute(1, 2, 0).numpy()

                    colormap = cv2.COLORMAP_JET
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), colormap)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    heatmap = np.float32(heatmap) / 255

                    image_weight = 0.5
                    cam = (1 - image_weight) * heatmap + image_weight * img_np
                    cam = cam / np.max(cam)

                    ax.axis('off')
                    ax.set_title(f'CAM classifier {sub_row_idx - 1}')

                    ax.imshow(cam, interpolation='lanczos')

    plt.tight_layout()
    fig.savefig(filename)

    if wandb_name is not None:
        wandb.log({wandb_name: fig}, step=wandb_step)

    plt.close(fig)

def plot_attention(original_image, segmentation, words_attention_masks, filename, wandb_name=None, wandb_step=None):
    scale_factor = 4.0
    num_cols = 2 + len(words_attention_masks)
    num_rows = 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))

    for i in range(0, num_cols):
        if i == 0:
            img = original_image.cpu().detach()
            title = 'original'
        elif i == 1:
            img = segmentation.cpu().detach()
            title = 'segmentation'
        else:
            word = list(words_attention_masks.keys())[i-2]
            img = words_attention_masks[word].cpu().detach()
            title = word

        img = torch.clamp(img.permute(1, 2, 0), min=0.0, max=1.0)

        ax = axs[i]
        ax.axis('off')
        ax.set_title(title)
        if i < 2:
            ax.imshow(img, interpolation='lanczos')
        else:
            ax.imshow(img, cmap='viridis', interpolation='lanczos')

    plt.tight_layout()
    fig.savefig(filename)

    if wandb_name is not None:
        wandb.log({wandb_name: fig}, step=wandb_step)


    plt.close(fig)
