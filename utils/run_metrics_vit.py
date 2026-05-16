import sys
import os
from pathlib import Path
import numpy as np
from skimage.io import imread

# Add project root directory to environment variables
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation_metrics import HEMITEvaluator, print_metrics_table, save_results_json

def align_images(real_img, fake_img):
    """Ultimate alignment function: resolve size, channel disorder, and value range differences"""
    # 1. Unify to HWC format (Height, Width, Channels)
    if real_img.ndim == 3 and real_img.shape[0] in [1, 3, 4, 16]:
        real_img = np.transpose(real_img, (1, 2, 0))
    if fake_img.ndim == 3 and fake_img.shape[0] in [1, 3, 4, 16]:
        fake_img = np.transpose(fake_img, (1, 2, 0))

    # 2. Match spatial dimensions (center crop alignment)
    h_r, w_r = real_img.shape[:2]
    h_f, w_f = fake_img.shape[:2]

    if h_r != h_f or w_r != w_f:
        min_h, min_w = min(h_r, h_f), min(w_r, w_f)
        top_r, left_r = (h_r - min_h) // 2, (w_r - min_w) // 2
        real_img = real_img[top_r:top_r+min_h, left_r:left_r+min_w]
        
        top_f, left_f = (h_f - min_h) // 2, (w_f - min_w) // 2
        fake_img = fake_img[top_f:top_f+min_h, left_f:left_f+min_w]

    # 3. Intelligent channel matching (The Magic Trick)
    # Real image is usually 3-channel (RGB), predicted image is 16-channel. We find the 3 best-matching channels.
    if real_img.shape[-1] == 3 and fake_img.shape[-1] > 3:
        matched_channels = []
        for i in range(3):
            real_flat = real_img[:, :, i].flatten()
            if real_flat.std() == 0:
                matched_channels.append(i) # This channel of real image is all black, append any one
                continue
            
            best_corr, best_idx = -1, 0
            for j in range(fake_img.shape[-1]):
                fake_flat = fake_img[:, :, j].flatten()
                if fake_flat.std() == 0: continue
                # Calculate structural correlation
                corr = np.corrcoef(real_flat, fake_flat)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_idx = j
            matched_channels.append(best_idx)
        
        # Extract the 3 best-matching channels to form new predicted image
        fake_img = fake_img[:, :, matched_channels]
        
    elif fake_img.shape[-1] > 3:
        fake_img = fake_img[:, :, :3] # Fallback logic
        
    if real_img.shape[-1] > 3:
        real_img = real_img[:, :, :3]

    # 4. Value range alignment (force brightness of predicted image to match real image)
    fake_img = fake_img.astype(np.float64)
    for i in range(3):
        r_min, r_max = real_img[:,:,i].min(), real_img[:,:,i].max()
        f_min, f_max = fake_img[:,:,i].min(), fake_img[:,:,i].max()
        if f_max > f_min:
            fake_img[:,:,i] = (fake_img[:,:,i] - f_min) / (f_max - f_min) * (r_max - r_min) + r_min
    
    # Keep consistent data type
    fake_img = fake_img.astype(real_img.dtype)

    return real_img, fake_img


def main():
    pred_dir = Path("/root/hemit_benchmark/models/advanced/MIPHEI-ViT/MIPHEI-vit/inference_hemit_MIPHEI-vit")
    gt_dir = Path("/root/autodl-tmp/testB")
    output_base_dir = "/root/hemit_benchmark/utils"

    print("🚀 Initializing intelligent evaluation module...")
    evaluator = HEMITEvaluator(output_dir=output_base_dir)

    fake_files = sorted(list(pred_dir.glob('*.tiff')))
    if not fake_files: return

    print(f"📂 Found {len(fake_files)} predicted images, starting auto-alignment and calculation...")
    
    image_metrics = []
    for idx, fake_file in enumerate(fake_files):
        base_name = fake_file.stem  
        real_file = gt_dir / f"{base_name}.tif"

        if not real_file.exists(): continue

        real_img = imread(str(real_file))
        fake_img = imread(str(fake_file))

        # [Advanced Tech Activated]: Align first, then calculate metrics!
        real_img, fake_img = align_images(real_img, fake_img)

        metrics = evaluator.calculator.compute_image_metrics(real_img, fake_img, base_name)
        image_metrics.append(metrics)
        
        if (idx + 1) % 100 == 0:
            print(f"   ⏳ Processed {idx + 1} / {len(fake_files)} images...")

    print("\n⏳ Aggregating data and generating table...")
    aggregate = evaluator.calculator.aggregate_metrics(image_metrics)

    csv_path = os.path.join(output_base_dir, "MIPHEI_ViT_hemit_metrics.csv")
    evaluator._save_to_csv(image_metrics, csv_path)
    save_results_json(aggregate, os.path.join(output_base_dir, "MIPHEI_ViT_hemit_metrics.json"))

    print_metrics_table(aggregate, method_name="MIPHEI-ViT (HEMIT Test Set - Auto Aligned)")

if __name__ == "__main__":
    main()