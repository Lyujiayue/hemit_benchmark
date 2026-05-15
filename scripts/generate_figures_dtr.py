```python
import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from tqdm import tqdm

def prepare_tensor(img_path):
    """Read image and convert to Tensor format (C, H, W) required by torchvision, normalize to [0,1]"""
    # Use imread to process .tif or .png files
    img = imread(str(img_path))
    # Ensure it is RGB
    if len(img.shape) == 2: # Convert grayscale to RGB
        img = img[:, :, np.newaxis].repeat(3, axis=2)
    elif img.shape[2] == 4: # Convert RGBA to RGB
        img = img[:, :, :3]
        
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor

def main():
    # ======= 1. Path Configuration =======
    # DTR prediction image path
    pred_dir = Path("/root/hemit_benchmark/models/advanced/dtr/dtr_preds")
    # Original dataset path (used to find Input and Ground Truth)
    input_dir = Path("/root/autodl-tmp/test/input")
    label_dir = Path("/root/autodl-tmp/test/label")
    
    output_dir = Path("/root/hemit_benchmark/report_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======= 2. Get prediction files and select the first 8 for display =======
    # Find all .tif or .png files
    all_preds = sorted([f for f in pred_dir.iterdir() if f.suffix in ['.tif', '.png', '.jpg']])
    
    if not all_preds:
        print(f"❌ Error: No prediction images found in {pred_dir}!")
        return

    # Select the first 8 for display
    selected_preds = all_preds[:8]

    inputs_list, fakes_list, reals_list = [], [], []

    # ======= 3. Match images across folders =======
    print("🔍 Matching images from different directories...")
    for pred_path in tqdm(selected_preds):
        base_name = pred_path.name # File name, e.g., 137_1855.png
        
        # Corresponding input and label paths
        in_path = input_dir / base_name
        gt_path = label_dir / base_name
        
        if in_path.exists() and gt_path.exists():
            try:
                inputs_list.append(prepare_tensor(in_path))
                fakes_list.append(prepare_tensor(pred_path))
                reals_list.append(prepare_tensor(gt_path))
            except Exception as e:
                print(f"⚠️ Error loading {base_name}: {e}")
        else:
            print(f"⚠️ Missing pairs for {base_name}, skipping.")

    if not inputs_list:
        print("❌ Error: No image groups loaded successfully, please check if the filenames match.")
        return

    # ======= 4. Stack into Batch Tensors =======
    inputs_tensor = torch.stack(inputs_list)
    fakes_tensor = torch.stack(fakes_list)
    reals_tensor = torch.stack(reals_list)

    # ======= 5. Create grid (2x4 layout) =======
    grid_inputs = vutils.make_grid(inputs_tensor, nrow=4, padding=4)
    grid_fakes = vutils.make_grid(fakes_tensor, nrow=4, padding=4)
    grid_reals = vutils.make_grid(reals_tensor, nrow=4, padding=4)

    # ======= 6. Assemble large image using Matplotlib =======
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.05)

    # Set titles (corresponding to DTR task)
    titles = ['Input (H&E Image)', 'Fake (Predicted mIHC)', 'Real (Ground Truth)']
    grids = [grid_inputs, grid_fakes, grid_reals]

    for i in range(3):
        img_np = grids[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(titles[i], fontsize=18, pad=10)
        axes[i].axis('off')

    # ======= 7. Save high-definition large image =======
    output_path = output_dir / "DTR_PaperStyle_Comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Success! DTR layout saved to: {output_path}")

if __name__ == "__main__":
    main()
```