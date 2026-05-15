import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as functional
import Attention_GAN
from reglib import reg

# ======= 1. 环境与路径配置 =======
# 请确保这些路径在你的 AutoDL 实例中是正确的
DATA_ROOT = "/root/autodl-tmp/test"
HE_DIR = os.path.join(DATA_ROOT, 'input')    # 输入 H&E 图像
PAS_DIR = os.path.join(DATA_ROOT, 'label')   # 目标 PAS 图像 (用于计算配准)
SAVE_DIR = "/root/hemit_benchmark/models/advanced/dtr/dtr_preds"

# 你的单一权重文件路径
CKPT_PATH = "/root/hemit_benchmark/models/advanced/dtr/checkpoints/hemit_weight.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_inference():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ======= 2. 初始化网络架构 (参考官方截图参数) =======
    # 生成器
    netG = Attention_GAN.Generator(
        n_channels=64, in_channels=3, out_channels=3, 
        batch_norm=False, padding=1, pooling_mode="maxpool"
    ).to(DEVICE)

    # 配准模块 (由于权重文件里没有它们，这里暂时只能初始化)
    RegGT = reg.Reg(128, 128, 3, 3, DEVICE, True).to(DEVICE)
    spatial_transform = reg.Transformer_2D()

    # ======= 3. 加载权重 =======
    print(f"📦 Loading Generator weights from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    
    # 你的文件是扁平的 state_dict，直接加载给 netG
    try:
        netG.load_state_dict(ckpt)
        print("✅ Generator weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        # 如果报错，尝试兼容模式（比如有些保存带 'module.' 前缀）
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        netG.load_state_dict(new_ckpt)
        print("✅ Generator weights loaded with prefix filtering.")

    netG.eval()
    RegGT.eval() # 即使是随机的，也需要 eval 模式

    # 获取测试集文件列表
    img_names = sorted([f for f in os.listdir(HE_DIR) if f.endswith(('.png', '.tif', '.jpg'))])

    print(f"🚀 Starting DTR line inference on {len(img_names)} images...")
    
    with torch.no_grad():
        for name in tqdm(img_names):
            # 1. 路径准备
            input_path = os.path.join(HE_DIR, name)
            target_path = os.path.join(PAS_DIR, name)
            
            if not os.path.exists(target_path):
                print(f"⚠️ Missing label for {name}, skipping...")
                continue

            # 2. 读取与图像预处理 (官方逻辑)
            input_img = Image.open(input_path).convert('RGB')
            target_img = Image.open(target_path).convert('RGB')

            # 官方使用 Center Crop 到 128x128
            input_img = functional.center_crop(input_img, (128, 128))
            target_img = functional.center_crop(target_img, (128, 128))

            # 归一化到 [-1, 1] 范围: $tensor = (pixel / 255 - 0.5) * 2$
            input_tensor = (functional.to_tensor(input_img) - 0.5) * 2
            target_tensor = (functional.to_tensor(target_img) - 0.5) * 2
            
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
            target_tensor = target_tensor.unsqueeze(0).to(DEVICE)

            # ======= 4. DTR 推理流水线 (参考图 4) =======
            # Step A: 生成原始预测图
            rec = netG(input_tensor)
            
            # Step B: 计算配准 Mesh (此步目前依赖初始化参数)
            # DTR 的核心：将生成的图像 rec 向目标图像 target 进行几何对齐
            mesh = RegGT(rec, target_tensor)
            
            # Step C: 执行空间变换得到最终结果
            rec_reg, _ = spatial_transform(rec, mesh)

            # ======= 5. 后处理与保存 =======
            # 反归一化到 [0, 255]: $pixel = (tensor + 1) / 2 * 255$
            out_np = ((rec_reg[0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype('uint8')
            out_np = np.transpose(out_np, (1, 2, 0)) # 转换回 HWC 格式
            
            final_img = Image.fromarray(out_np)
            final_img.save(os.path.join(SAVE_DIR, name))

    print(f"✨ Inference complete. Files saved to: {SAVE_DIR}")

if __name__ == "__main__":
    run_inference()