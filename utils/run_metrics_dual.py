import os
# 因为 run_metrics_dual.py 和 evaluation_metrics.py 在同一个 utils 文件夹下，直接导入即可
from evaluation_metrics import HEMITEvaluator, print_metrics_table

def main():
    # 1. 设置输出 CSV 和结果的目录（建议放在工程根目录的 metrics_output 下保持整洁）
    output_dir = "/root/hemit_benchmark/metrics_output"
    os.makedirs(output_dir, exist_ok=True)
    evaluator = HEMITEvaluator(output_dir=output_dir)

    # 2. 指定预测结果所在的目录
    # 注意：Dual-branch 跑完后，real_B(真实标签)和 fake_B(预测图)都在这个文件夹里
    results_dir = "/root/hemit_benchmark/models/baselines/dual_branch_pix2pix/results/pretrained_hemit/test_latest/images"

    print("🚀 开始读取图像并计算指标 (SSIM, Pearson R, PSNR)...")
    
    # 3. 运行评测
    image_metrics, aggregate = evaluator.evaluate_from_directory(
        real_dir=results_dir,  # 找 real_B 
        fake_dir=results_dir,  # 找 fake_B
        output_csv=os.path.join(output_dir, "dual_branch_results.csv")
    )

    # 4. 打印可以复制到报告里的 Markdown 表格
    print_metrics_table(aggregate, method_name="Official Dual-Branch (Pretrained)")
    print(f"✅ 评测完成！详细结果已保存至: {output_dir}/dual_branch_results.csv")

if __name__ == "__main__":
    main()