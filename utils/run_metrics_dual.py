import os
# Since run_metrics_dual.py and evaluation_metrics.py are in the same utils folder, direct import is supported
from evaluation_metrics import HEMITEvaluator, print_metrics_table

def main():
    # 1. Set the directory for output CSV and results (recommended to be placed under metrics_output in the project root for cleanliness)
    output_dir = "/root/hemit_benchmark/metrics_output"
    os.makedirs(output_dir, exist_ok=True)
    evaluator = HEMITEvaluator(output_dir=output_dir)

    # 2. Specify the directory where the prediction results are located
    # Note: After Dual-branch runs, both real_B (ground truth labels) and fake_B (predictions) are in this folder
    results_dir = "/root/hemit_benchmark/models/baselines/dual_branch_pix2pix/results/pretrained_hemit/test_latest/images"

    print("🚀 Starting to read images and calculate metrics (SSIM, Pearson R, PSNR)...")
    
    # 3. Run evaluation
    image_metrics, aggregate = evaluator.evaluate_from_directory(
        real_dir=results_dir,  # Look for real_B
        fake_dir=results_dir,  # Look for fake_B
        output_csv=os.path.join(output_dir, "dual_branch_results.csv")
    )

    # 4. Print a Markdown table that can be copied directly into the report
    print_metrics_table(aggregate, method_name="Official Dual-Branch (Pretrained)")
    print(f"✅ Evaluation complete! Detailed results saved to: {output_dir}/dual_branch_results.csv")

if __name__ == "__main__":
    main()