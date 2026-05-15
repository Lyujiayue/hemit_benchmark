```python
import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path

def prepare_tensor(img_path):
    """Read an image and convert it to the Tensor format required by torchvision (C, H, W), normalized to [0,1]"""
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
    # iterdir() directly traverses the folder, bypassing glob's special parsing of []
    all_files = list(results_dir.iterdir())
    all_fakes = sorted([f for f in all_files if 'fake_B' in f.name])
    
    selected_bases = []
    # Get the first 8 base_names
    for f in all_fakes[:8]:
        ext = f.suffix
        if '_fake_B' in f.name:
            base = f.name.replace(f'_fake_B{ext}', '')
        else:
            base = f.name.replace('fake_B_', '').replace(ext, '')
        selected_bases.append(base)

    if len(selected_bases) < 8:
        print(f"⚠️ Warning: Only found {len(selected_bases)} image groups, the collage may be incomplete!")

    inputs_list, fakes_list, reals_list = [], [], []

    # 3. Read all images and store them by category (using pure string matching to prevent [] errors)
    for base_name in selected_bases:
        real_a_path, fake_b_path, real_b_path = None, None, None
        
        for f in all_files:
            if base_name in f.name:
                if 'real_A' in f.name: real_a_path = f
                if 'fake_B' in f.name: fake_b_path = f
                if 'real_B' in f.name: real_b_path = f
                
        if real_a_path and fake_b_path and real_b_path:
            inputs_list.append(prepare_tensor(real_a_path))
            fakes_list.append(prepare_tensor(fake_b_path))
            reals_list.append(prepare_tensor(real_b_path))
        else:
            print(f"Could not find the complete set of three images for {base_name}, skipping.")

    if not inputs_list:
        print("❌ Error: No images loaded successfully, please check the path.")
        return

    # 4. Stack the lists into Batch Tensors
    inputs_tensor = torch.stack(inputs_list)
    fakes_tensor = torch.stack(fakes_list)
    reals_tensor = torch.stack(reals_list)

    # 5. Create 2x4 grid images with black borders
    grid_inputs = vutils.make_grid(inputs_tensor, nrow=4, padding=4)
    grid_fakes = vutils.make_grid(fakes_tensor, nrow=4, padding=4)
    grid_reals = vutils.make_grid(reals_tensor, nrow=4, padding=4)

    # 6. Assemble the final large image using Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.02)

    titles = ['Input (H&E Input)', 'Fake (Predicted mIHC)', 'Real (Ground Truth)']
    grids = [grid_inputs, grid_fakes, grid_reals]

    for i in range(3):
        img_np = grids[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(titles[i], fontsize=18, pad=10)
        axes[i].axis('off')

    # 7. Save high-definition large image
    output_path = output_dir / "DualBranch_PaperStyle_Comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Success! Top-tier paper layout style large image saved to: {output_path}")

if __name__ == "__main__":
    main()