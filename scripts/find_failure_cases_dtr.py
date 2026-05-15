import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

def prepare_tensor(img_path):
    """读取图像并转换为 torchvision 要求的 Tensor 格式 (C, H, W)，归一化到 [0,1]"""
    img = imread(str(img_path))
    # 确保是 RGB
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis].repeat(3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
        
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor

def main():
    # ======= 1. 路径设置 =======
    pred_dir = Path("/root/hemit_benchmark/models/advanced/dtr/dtr_preds")
    input_dir = Path("/root/autodl-tmp/test/input")
    label_dir = Path("/root/autodl-tmp/test/label")
    
    output_dir = Path("/root/hemit_benchmark/report_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======= 2. 获取所有预测文件 =======
    all_preds = sorted([f for f in pred_dir.iterdir() if f.suffix in ['.tif', '.png', '.jpg']])
    
    if not all_preds:
        print(f"❌ Error: 在 {pred_dir} 中没找到预测图片！")
        return

    # ======= 3. 遍历全量测试集，寻找误差最大的 Failure Cases =======
    print("🔍 Scanning the full DTR test set to find Failure Cases (Largest L1 Errors)...")
    error_list = []
    
    for pred_path in tqdm(all_preds):
        base_name = pred_path.name
        in_path = input_dir / base_name
        gt_path = label_dir / base_name
        
        
        if gt_path.exists():
            # 读取图片 (注意这里改成了 real_img_raw)
            fake_img = imread(str(pred_path)).astype(np.float32)
            real_img_raw = imread(str(gt_path)).astype(np.float32) 
            
            # 检查尺寸并缩放标签图
            if fake_img.shape != real_img_raw.shape:
                real_img = resize(real_img_raw, (128, 128), anti_aliasing=True, preserve_range=True)
            else:
                real_img = real_img_raw
            
            # 计算 L1 平均绝对误差
            l1_error = np.mean(np.abs(fake_img - real_img))
            
            error_list.append({
                'base_name': base_name,
                'loss': l1_error,
                'input_path': in_path,
                'pred_path': pred_path,
                'gt_path': gt_path
            })

    # ======= 4. 排序并提取误差最大的前 4 名 =======
    error_list.sort(key=lambda x: x['loss'], reverse=True)
    worst_4_cases = error_list[:4]
    
    print("\n⚠️ Found the top 4 DTR Failure Cases:")
    for i, item in enumerate(worst_4_cases):
        print(f"Rank {i+1} | File: {item['base_name']} | L1 Error: {item['loss']:.2f}")

    # ======= 5. 读取这 4 组图像进行拼图 =======
    inputs_list, fakes_list, reals_list = [], [], []
    for item in worst_4_cases:
        inputs_list.append(prepare_tensor(item['input_path']))
        fakes_list.append(prepare_tensor(item['pred_path']))
        reals_list.append(prepare_tensor(item['gt_path']))

    # 转换为 Batch Tensors
    inputs_tensor = torch.stack(inputs_list)
    fakes_tensor = torch.stack(fakes_list)
    reals_tensor = torch.stack(reals_list)

    # ======= 6. 创建网格图像 (4张图排成 1行 4列) =======
    grid_inputs = vutils.make_grid(inputs_tensor, nrow=4, padding=4)
    grid_fakes = vutils.make_grid(fakes_tensor, nrow=4, padding=4)
    grid_reals = vutils.make_grid(reals_tensor, nrow=4, padding=4)

    # ======= 7. 使用 Matplotlib 组装大图 =======
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    plt.subplots_adjust(wspace=0.02)

    # 标题设置（Failure Pred 部分标红）
    titles = ['Input (H&E Input)', 'DTR Failure Pred (mIHC)', 'Real (Ground Truth)']
    grids = [grid_inputs, grid_fakes, grid_reals]

    for i in range(3):
        img_np = grids[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        # 将中间的预测失败图标题设为深红色
        title_color = 'darkred' if i == 1 else 'black'
        axes[i].set_title(titles[i], fontsize=18, fontweight='bold', color=title_color, pad=10)
        axes[i].axis('off')

    # ======= 8. 保存结果 =======
    output_path = output_dir / "DTR_FailureCases_Comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Success! DTR failure case analysis saved to: {output_path}")

if __name__ == "__main__":
    main()