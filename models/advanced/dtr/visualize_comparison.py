import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 路径配置
he_dir = "/root/autodl-tmp/test/input"
pred_dir = "/root/hemit_benchmark/models/advanced/dtr/dtr_preds"
gt_dir = "/root/autodl-tmp/test/label"
save_path = "/root/hemit_benchmark/models/advanced/dtr/comparison_check.png"

def visualize_samples(num_samples=4):
    filenames = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.tif'))])
    samples = random.sample(filenames, num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i, name in enumerate(samples):
        # 读取三张图
        he_img = Image.open(os.path.join(he_dir, name)).convert('RGB')
        pred_img = Image.open(os.path.join(pred_dir, name))
        gt_img = Image.open(os.path.join(gt_dir, name))
        
        # 统一尺寸显示
        axes[i, 0].imshow(he_img)
        axes[i, 0].set_title(f"H&E: {name}")
        
        axes[i, 1].imshow(pred_img)
        axes[i, 1].set_title("DTR Prediction")
        
        axes[i, 2].imshow(gt_img)
        axes[i, 2].set_title("Ground Truth (mIHC)")
        
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Comparison saved to: {save_path}")

if __name__ == "__main__":
    visualize_samples()