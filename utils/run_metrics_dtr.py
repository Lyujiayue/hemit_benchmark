"""
Reproduction script for DGR/DTR Benchmark.
This script focuses solely on calculating and printing the SSIM, Pearson R, and PSNR 
metrics for the HEMIT dataset using the official evaluation_metrics.py module.

Location: /root/hemit_benchmark/utils/run_metrics_dtr.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

# Ensure the utility folder is in the system path for importing evaluation_metrics.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # Importing the official logic from evaluation_metrics.py
    from evaluation_metrics import MetricsCalculator, print_metrics_table, save_results_json
except ImportError:
    print("❌ Error: evaluation_metrics.py not found in the /root/hemit_benchmark/utils/ directory.")
    sys.exit(1)

# ======= Configuration =======
# The directory where DTR generated images are stored
PRED_DIR = "/root/hemit_benchmark/models/advanced/dtr/dtr_preds"
# The directory containing the ground truth mIHC labels
GT_DIR = "/root/autodl-tmp/test/label"
# The destination for saving the metric results
REPORT_DIR = "/root/hemit_benchmark/models/advanced/dtr/metrics_reports"

def run_benchmark():
    # 1. Initialize result directories and MetricsCalculator
    os.makedirs(REPORT_DIR, exist_ok=True)
    calc = MetricsCalculator()
    image_metrics_list = []

    pred_path = Path(PRED_DIR)
    gt_path = Path(GT_DIR)
    
    if not pred_path.exists():
        print(f"❌ Error: Predicted images directory not found at {PRED_DIR}")
        return

    # 2. Collect all predicted filenames
    # We use files in PRED_DIR as the reference for evaluation
    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith(('.png', '.jpg', '.tif'))])
    
    print(f"🧪 Evaluating {len(pred_files)} samples from DGR/DTR reproduction...")

    # 3. Main Evaluation Loop
    for filename in tqdm(pred_files, desc="Processing Images"):
        p_file = pred_path / filename
        g_file = gt_path / filename

        if not g_file.exists():
            print(f"⚠️ Warning: Corresponding ground truth for {filename} missing in {GT_DIR}. Skipping.")
            continue

        # Load images (H, W, C)
        real_img = imread(str(g_file))
        fake_img = imread(str(p_file))

        # IMPORTANT: DTR usually outputs at 128x128. 
        # Resize prediction to match Ground Truth dimensions to ensure metric accuracy.
        if real_img.shape[:2] != fake_img.shape[:2]:
            fake_img = (resize(fake_img, real_img.shape, 
                               anti_aliasing=True, 
                               preserve_range=True)).astype(np.uint8)

        # Compute metrics for all 3 channels (DAPI, panCK, CD3)
        metrics = calc.compute_image_metrics(real_img, fake_img, filename)
        image_metrics_list.append(metrics)

    if not image_metrics_list:
        print("❌ No images were processed. Please check if file names in 'input' and 'label' match.")
        return

    # 4. Aggregation and Output
    # This step uses the official safety-patched logic for mean and std calculation
    aggregate = calc.aggregate_metrics(image_metrics_list)

    # Print the final benchmark table to the console as requested
    print_metrics_table(aggregate, method_name="DGR/DTR Reproduction Benchmark")

    # 5. Persistent storage of results
    save_results_json(aggregate, os.path.join(REPORT_DIR, "dtr_summary_metrics.json"))
    
    print(f"✅ Benchmark finished. Summary saved to: {REPORT_DIR}")

if __name__ == "__main__":
    run_benchmark()