import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from pathlib import Path

def align_images(real_img, fake_img):
    """Align ViT 16-channel, CHW format with ground truth image"""
    if real_img.ndim == 3 and real_img.shape[0] in [1, 3, 4, 16]:
        real_img = np.transpose(real_img, (1, 2, 0))
    if fake_img.ndim == 3 and fake_img.shape[0] in [1, 3, 4, 16]:
        fake_img = np.transpose(fake_img, (1, 2, 0))

    h_r, w_r = real_img.shape[:2]
    h_f, w_f = fake_img.shape[:2]
    if h_r != h_f or w_r != w_f:
        min_h, min_w = min(h_r, h_f), min(w_r, w_f)
        top_r, left_r = (h_r - min_h) // 2, (w_r - min_w) // 2
        real_img = real_img[top_r:top_r+min_h, left_r:left_r+min_w]
        top_f, left_f = (h_f - min_h) // 2, (w_f - min_w) // 2
        fake_img = fake_img[top_f:top_f+min_h, left_f:left_f+min_w]

    if real_img.shape[-1] == 3 and fake_img.shape[-1] > 3:
        matched_channels = []
        for i in range(3):
            real_flat = real_img[:, :, i].flatten()
            if real_flat.std() == 0:
                matched_channels.append(i)
                continue
            best_corr, best_idx = -1, 0
            for j in range(fake_img.shape[-1]):
                fake_flat = fake_img[:, :, j].flatten()
                if fake_flat.std() == 0: continue
                corr = np.corrcoef(real_flat, fake_flat)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_idx = j
            matched_channels.append(best_idx)
        fake_img = fake_img[:, :, matched_channels]
    elif fake_img.shape[-1] > 3:
        fake_img = fake_img[:, :, :3]
        
    if real_img.shape[-1] > 3:
        real_img = real_img[:, :, :3]

    fake_img = fake_img.astype(np.float64)
    for i in range(3):
        r_min, r_max = real_img[:,:,i].min(), real_img[:,:,i].max()
        f_min, f_max = fake_img[:,:,i].min(), fake_img[:,:,i].max()
        if f_max > f_min:
            fake_img[:,:,i] = (fake_img[:,:,i] - f_min) / (f_max - f_min) * (r_max - r_min) + r_min
            
    return real_img, fake_img.astype(real_img.dtype)

def center_crop(img, target_h, target_w):
    """Crop H&E input image to match the size of the predicted image"""
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return img[top:top+target_h, left:left+target_w]

def prepare_tensor_from_numpy(img):
    """Convert numpy image to Tensor required by torchvision"""
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

def main():
    # 1. Auto-adapt all paths
    pred_dir = Path("/root/hemit_benchmark/models/advanced/MIPHEI-ViT/MIPHEI-vit/inference_hemit_MIPHEI-vit")
    gt_dir = Path("/root/autodl-tmp/testB")
    
    # Find H&E input path (compatible with testA or test/input)
    input_dir = Path("/root/autodl-tmp/testA")
    if not input_dir.exists() or len(list(input_dir.glob('*.tif'))) == 0:
        input_dir = Path("/root/autodl-tmp/test/input")

    output_dir = Path("/root/hemit_benchmark/report_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Get first 8 predicted images
    fake_files = sorted(list(pred_dir.glob('*.tiff')))[:8]
    if len(fake_files) < 8:
        print(f"⚠️ Warning: Only found {len(fake_files)} predicted images, collage may be incomplete!")

    inputs_list, fakes_list, reals_list = [], [], []

    # 3. Read, align and convert to Tensor
    for fake_file in fake_files:
        base_name = fake_file.stem
        real_file = gt_dir / f"{base_name}.tif"
        input_file = input_dir / f"{base_name}.tif"

        if not real_file.exists() or not input_file.exists():
            print(f"Complete image set not found for {base_name}, skipping.")
            continue

        real_img = imread(str(real_file))
        fake_img = imread(str(fake_file))
        input_img = imread(str(input_file))

        # Core: Auto align predicted image with ground truth
        real_aligned, fake_aligned = align_images(real_img, fake_img)

        # Crop and extract H&E input (remove Alpha channel if exists)
        if input_img.ndim == 3 and input_img.shape[-1] > 3:
            input_img = input_img[..., :3]
        input_aligned = center_crop(input_img, real_aligned.shape[0], real_aligned.shape[1])

        inputs_list.append(prepare_tensor_from_numpy(input_aligned))
        fakes_list.append(prepare_tensor_from_numpy(fake_aligned))
        reals_list.append(prepare_tensor_from_numpy(real_aligned))

    if not inputs_list:
        print("❌ Error: No images loaded successfully, please check the paths.")
        return

    # 4. Stack into Batch Tensors
    inputs_tensor = torch.stack(inputs_list)
    fakes_tensor = torch.stack(fakes_list)
    reals_tensor = torch.stack(reals_list)

    # 5. Generate 2x4 grid with black padding
    grid_inputs = vutils.make_grid(inputs_tensor, nrow=4, padding=4)
    grid_fakes = vutils.make_grid(fakes_tensor, nrow=4, padding=4)
    grid_reals = vutils.make_grid(reals_tensor, nrow=4, padding=4)

    # 6. Assemble final large image using Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.02)

    titles = ['Input (H&E Input)', 'Fake (MIPHEI-ViT Predicted)', 'Real (Ground Truth)']
    grids = [grid_inputs, grid_fakes, grid_reals]

    for i in range(3):
        img_np = grids[i].permute(1, 2, 0).numpy()
        # Prevent out-of-bounds
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].set_title(titles[i], fontsize=18, pad=10)
        axes[i].axis('off')

    # 7. Save high-definition large image
    output_path = output_dir / "MIPHEI_ViT_PaperStyle_Comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Done! MIPHEI-ViT top-tier paper-style large image saved to: {output_path}")

if __name__ == "__main__":
    main()