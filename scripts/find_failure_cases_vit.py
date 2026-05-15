import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
import numpy as np
from tqdm import tqdm

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
    h, w = img.shape[:2]
    if h == target_h and w == target_w: return img
    top, left = (h - target_h) // 2, (w - target_w) // 2
    return img[top:top+target_h, left:left+target_w]

def prepare_tensor_from_numpy(img):
    img = img.astype(np.float32)
    if img.max() > 1.0: img = img / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

def main():
    # 1. Path Configuration
    pred_dir = Path("/root/hemit_benchmark/models/advanced/MIPHEI-ViT/MIPHEI-vit/inference_hemit_MIPHEI-vit")
    gt_dir = Path("/root/autodl-tmp/testB")
    
    input_dir = Path("/root/autodl-tmp/testA")
    if not input_dir.exists() or len(list(input_dir.glob('*.tif'))) == 0:
        input_dir = Path("/root/autodl-tmp/test/input")

    output_dir = Path("/root/hemit_benchmark/report_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fake_files = sorted(list(pred_dir.glob('*.tiff')))
    if not fake_files: return

    # 2. Traverse and calculate L1 error for all images (alignment required first)
    print("🔍 Scanning the full test set to find Failure Cases with maximum error... (takes about 1-2 minutes)")
    error_list = []
    
    for fake_file in tqdm(fake_files):
        base_name = fake_file.stem
        real_file = gt_dir / f"{base_name}.tif"
        if not real_file.exists(): continue

        real_img = imread(str(real_file))
        fake_img = imread(str(fake_file))
        
        # Error calculation is meaningful only after alignment and channel matching
        real_aligned, fake_aligned = align_images(real_img, fake_img)
        
        # Calculate L1 absolute error
        l1_error = np.mean(np.abs(fake_aligned.astype(np.float32) - real_aligned.astype(np.float32)))
        error_list.append({'base_name': base_name, 'loss': l1_error})

    # 3. Sort and extract top 4 with maximum error
    error_list.sort(key=lambda x: x['loss'], reverse=True)
    worst_4_cases = error_list[:4]
    
    print("\n🚨 Found the top 4 Failure Cases with maximum error:")
    for i, item in enumerate(worst_4_cases):
        print(f"Top {i+1} | File: {item['base_name']} | L1 Error: {item['loss']:.2f}")

    # 4. Reload these 4 groups of images for collage
    inputs_list, fakes_list, reals_list = [], [], []
    for item in worst_4_cases:
        base_name = item['base_name']
        real_file = gt_dir / f"{base_name}.tif"
        fake_file = pred_dir / f"{base_name}.tiff"
        input_file = input_dir / f"{base_name}.tif"

        real_img = imread(str(real_file))
        fake_img = imread(str(fake_file))
        input_img = imread(str(input_file))

        real_aligned, fake_aligned = align_images(real_img, fake_img)
        if input_img.ndim == 3 and input_img.shape[-1] > 3:
            input_img = input_img[..., :3]
        input_aligned = center_crop(input_img, real_aligned.shape[0], real_aligned.shape[1])

        inputs_list.append(prepare_tensor_from_numpy(input_aligned))
        fakes_list.append(prepare_tensor_from_numpy(fake_aligned))
        reals_list.append(prepare_tensor_from_numpy(real_aligned))

    inputs_tensor = torch.stack(inputs_list)
    fakes_tensor = torch.stack(fakes_list)
    reals_tensor = torch.stack(reals_list)

    # 5. Generate grid (1 row, 4 columns)
    grid_inputs = vutils.make_grid(inputs_tensor, nrow=4, padding=4)
    grid_fakes = vutils.make_grid(fakes_tensor, nrow=4, padding=4)
    grid_reals = vutils.make_grid(reals_tensor, nrow=4, padding=4)

    # 6. Assemble layout with Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    plt.subplots_adjust(wspace=0.02)

    titles = ['Input (H&E Input)', 'Fake (MIPHEI-ViT Failure Cases)', 'Real (Ground Truth)']
    grids = [grid_inputs, grid_fakes, grid_reals]

    for i in range(3):
        img_np = grids[i].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].set_title(titles[i], fontsize=18, pad=10)
        axes[i].axis('off')

    # 7. Save high-definition large image
    output_path = output_dir / "MIPHEI_ViT_FailureCases_Comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Done! MIPHEI-ViT failure case analysis large image saved to: {output_path}")

if __name__ == "__main__":
    main()