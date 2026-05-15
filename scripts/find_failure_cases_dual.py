import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
import numpy as np
from tqdm import tqdm

def prepare_tensor(img_path):
    """Read the image and convert it to the Tensor format required by torchvision (C, H, W) and normalize to [0,1]"""
    img = imread(str(img_path))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor

def main():
    # 1. Path setup
    results_dir = Path("/root/hemit_benchmark/models/baselines/dual_branch_pix2pix/results/pretrained_hemit/test_latest/images")
    output_dir = Path("/root/hemit_benchmark/report_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Get all prediction image files
    all_files = list(results_dir.iterdir())
    all_fakes = sorted([f for f in all_files if 'fake_B' in f.name])
    
    base_names = []
    for f in all_fakes:
        ext = f.suffix
        if '_fake_B' in f.name:
            base = f.name.replace(f'_fake_B{ext}', '')
        else:
            base = f.name.replace('fake_B_', '').replace(ext, '')
        base_names.append(base)

    # 3. Iterate and calculate L1 error (Mean Absolute Error)
    print("🔍 Scanning the full test set to find Failure Cases with the largest prediction errors...")
    error_list = []
    
    for base_name in tqdm(base_names):
        fake_b_path, real_b_path = None, None
        
        for f in all_files:
            if base_name in f.name:
                if 'fake_B' in f.name: fake_b_path = f
                if 'real_B' in f.name: real_b_path = f
                
        if fake_b_path and real_b_path:
            # Read as float32 to calculate error and avoid overflow
            fake_img = imread(str(fake_b_path)).astype(np.float32)
            real_img = imread(str(real_b_path)).astype(np.float32)
            
            # Calculate L1 absolute error (closer to 0 is better, larger is worse)
            l1_error = np.mean(np.abs(fake_img - real_img))
            
            error_list.append({
                'base_name': base_name,
                'loss': l1_error,
                'real_a_path': [f for f in all_files if base_name in f.name and 'real_A' in f.name][0],
                'fake_b_path': fake_b_path,
                'real_b_path': real_b_path
            })

    # 4. Sort and extract the top 4 with the largest errors
    error_list.sort(key=lambda x: x['loss'], reverse=True)
    worst_4_cases = error_list[:4]
    
    print("\n Found the 4 groups of Failure Cases with the largest errors:")
    for i, item in enumerate(worst_4_cases):
        print(f"Rank {i+1} | File: {item['base_name']} | L1 Error: {item['loss']:.2f}")

    # 5. Read these 4 groups of images for collage
    inputs_list, fakes_list, reals_list = [], [], []
    for item in worst_4_cases:
        inputs_list.append(prepare_tensor(item['real_a_path']))
        fakes_list.append(prepare_tensor(item['fake_b_path']))
        reals_list.append(prepare_tensor(item['real_b_path']))

    inputs_tensor = torch.stack(inputs_list)
    fakes_tensor = torch.stack(fakes_list)
    reals_tensor = torch.stack(reals_list)

    # 6. Create grid image (4 images, arranged in 1 row and 4 columns)
    grid_inputs = vutils.make_grid(inputs_tensor, nrow=4, padding=4)
    grid_fakes = vutils.make_grid(fakes_tensor, nrow=4, padding=4)
    grid_reals = vutils.make_grid(reals_tensor, nrow=4, padding=4)

    # 7. Assemble and typeset using Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(20, 4)) # Slightly lower the height to fit 1 row
    plt.subplots_adjust(wspace=0.02)

    titles = ['Input (H&E Input)', 'Fake (Predicted mIHC)', 'Real (Ground Truth)']
    grids = [grid_inputs, grid_fakes, grid_reals]

    for i in range(3):
        img_np = grids[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(titles[i], fontsize=18, pad=10)
        axes[i].axis('off')

    # 8. Save high-definition large image
    output_path = output_dir / "DualBranch_FailureCases_Comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Success! The failure case analysis large image has been saved to: {output_path}")

if __name__ == "__main__":
    main()