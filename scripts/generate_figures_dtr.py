import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from tqdm import tqdm

def prepare_tensor(img_path):
    """读取图像并转换为 torchvision 要求的 Tensor 格式 (C, H, W)，归一化到 [0,1]"""
    # 使用 imread 处理 .tif 或 .png
    img = imread(str(img_path))
    # 确保是 RGB
    if len(img.shape) == 2: # 灰度图转 RGB
        img = img[:, :, np.newaxis].repeat(3, axis=2)
    elif img.shape[2] == 4: # RGBA 转 RGB
        img = img[:, :, :3]
        
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor

def main():
    # ======= 1. 路径设置 =======
    # DTR 预测图路径
    pred_dir = Path("/root/hemit_benchmark/models/advanced/dtr/dtr_preds")
    # 原始数据集路径 (用于找 Input 和 Ground Truth)
    input_dir = Path("/root/autodl-tmp/test/input")
    label_dir = Path("/root/autodl-tmp/test/label")
    
    output_dir = Path("/root/hemit_benchmark/report_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======= 2. 获取预测文件并选择前 8 个作为展示 =======
    # 查找所有的 .tif 或 .png 文件
    all_preds = sorted([f for f in pred_dir.iterdir() if f.suffix in ['.tif', '.png', '.jpg']])
    
    if not all_preds:
        print(f"❌ Error: 在 {pred_dir} 中没找到预测图片！")
        return

    # 选择前 8 个进行展示
    selected_preds = all_preds[:8]

    inputs_list, fakes_list, reals_list = [], [], []

    # ======= 3. 跨文件夹匹配图像 =======
    print("🔍 Matching images from different directories...")
    for pred_path in tqdm(selected_preds):
        base_name = pred_path.name # 文件名，例如 137_1855.png
        
        # 对应的输入和标签路径
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
        print("❌ Error: 没有成功加载任何图像组，请检查文件名是否匹配。")
        return

    # ======= 4. 堆叠为 Batch Tensors =======
    inputs_tensor = torch.stack(inputs_list)
    fakes_tensor = torch.stack(fakes_list)
    reals_tensor = torch.stack(reals_list)

    # ======= 5. 创建网格 (2x4 布局) =======
    grid_inputs = vutils.make_grid(inputs_tensor, nrow=4, padding=4)
    grid_fakes = vutils.make_grid(fakes_tensor, nrow=4, padding=4)
    grid_reals = vutils.make_grid(reals_tensor, nrow=4, padding=4)

    # ======= 6. 使用 Matplotlib 拼装大图 =======
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.05)

    # 设置标题 (对应 DTR 任务)
    titles = ['Input (H&E Image)', 'DTR Prediction (PAS)', 'Ground Truth (PAS)']
    grids = [grid_inputs, grid_fakes, grid_reals]

    for i in range(3):
        img_np = grids[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(titles[i], fontsize=18, pad=15, fontweight='bold')
        axes[i].axis('off')

    # ======= 7. 保存高清大图 =======
    output_path = output_dir / "DTR_PaperStyle_Comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Success! DTR layout saved to: {output_path}")

if __name__ == "__main__":
    main()