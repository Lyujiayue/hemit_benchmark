import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as functional
import Attention_GAN
from reglib import reg

# ======= 1. Environment and Path Configuration =======
# Make sure these paths are correct on your AutoDL instance
DATA_ROOT = "/root/autodl-tmp/test"
HE_DIR = os.path.join(DATA_ROOT, 'input')    # Input H&E images
PAS_DIR = os.path.join(DATA_ROOT, 'label')   # Target PAS images (for registration calculation)
SAVE_DIR = "/root/hemit_benchmark/models/advanced/dtr/dtr_preds"

# Path to your single checkpoint file
CKPT_PATH = "/root/hemit_benchmark/models/advanced/dtr/checkpoints/hemit_weight.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_inference():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ======= 2. Initialize Network Architecture (parameters refer to official screenshot) =======
    # Generator
    netG = Attention_GAN.Generator(
        n_channels=64, in_channels=3, out_channels=3, 
        batch_norm=False, padding=1, pooling_mode="maxpool"
    ).to(DEVICE)

    # Registration module (only initialized temporarily as they are not in the checkpoint)
    RegGT = reg.Reg(128, 128, 3, 3, DEVICE, True).to(DEVICE)
    spatial_transform = reg.Transformer_2D()

    # ======= 3. Load Checkpoint Weights =======
    print(f"📦 Loading Generator weights from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    
    # Your file is a flat state_dict, load directly to netG
    try:
        netG.load_state_dict(ckpt)
        print("✅ Generator weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        # If error occurs, try compatibility mode (e.g., some checkpoints have 'module.' prefix)
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        netG.load_state_dict(new_ckpt)
        print("✅ Generator weights loaded with prefix filtering.")

    netG.eval()
    RegGT.eval() # Eval mode required even with random weights

    # Get test set file list
    img_names = sorted([f for f in os.listdir(HE_DIR) if f.endswith(('.png', '.tif', '.jpg'))])

    print(f"🚀 Starting DTR line inference on {len(img_names)} images...")
    
    with torch.no_grad():
        for name in tqdm(img_names):
            # 1. Path preparation
            input_path = os.path.join(HE_DIR, name)
            target_path = os.path.join(PAS_DIR, name)
            
            if not os.path.exists(target_path):
                print(f"⚠️ Missing label for {name}, skipping...")
                continue

            # 2. Image reading and preprocessing (official logic)
            input_img = Image.open(input_path).convert('RGB')
            target_img = Image.open(target_path).convert('RGB')

            # Official uses Center Crop to 128x128
            input_img = functional.center_crop(input_img, (128, 128))
            target_img = functional.center_crop(target_img, (128, 128))

            # Normalize to [-1, 1] range: $tensor = (pixel / 255 - 0.5) * 2$
            input_tensor = (functional.to_tensor(input_img) - 0.5) * 2
            target_tensor = (functional.to_tensor(target_img) - 0.5) * 2
            
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
            target_tensor = target_tensor.unsqueeze(0).to(DEVICE)

            # ======= 4. DTR Inference Pipeline (refer to Figure 4) =======
            # Step A: Generate raw prediction image
            rec = netG(input_tensor)
            
            # Step B: Calculate registration mesh (currently depends on initialized parameters)
            # Core of DTR: Geometrically align generated image rec to target image
            mesh = RegGT(rec, target_tensor)
            
            # Step C: Perform spatial transformation to get final result
            rec_reg, _ = spatial_transform(rec, mesh)

            # ======= 5. Post-processing and Saving =======
            # Denormalize to [0, 255]: $pixel = (tensor + 1) / 2 * 255$
            out_np = ((rec_reg[0].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype('uint8')
            out_np = np.transpose(out_np, (1, 2, 0)) # Convert back to HWC format
            
            final_img = Image.fromarray(out_np)
            final_img.save(os.path.join(SAVE_DIR, name))

    print(f"✨ Inference complete. Files saved to: {SAVE_DIR}")

if __name__ == "__main__":
    run_inference()