"""
Generate Final Report Images (Inference Script)
Runs inference on the test set and saves high-quality comparison grids.
"""
import os
import argparse
import yaml
import torch
import torchvision.utils as vutils
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import create_data_loaders
from models.advanced.dgr_dtr import create_dgr_model
from models.baselines.pix2pix import create_generator
from models.baselines.dual_branch import DualBranchGenerator

def denorm(t):
    """Restore images to the [0, 1] range for saving"""
    t = t.clone()
    for i in range(t.size(0)):
        ch = t[i]
        mn, mx = ch.min(), ch.max()
        if mx - mn > 1e-6:
            t[i] = (ch - mn) / (mx - mn)
    return t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the model config")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to best.pth")
    parser.add_argument('--data_root', type=str, required=True, help="Path to HEMIT data")
    parser.add_argument('--output_dir', type=str, default="report_figures", help="Where to save images")
    parser.add_argument('--num_samples', type=int, default=8, help="Number of images to generate (Assignment requires 8)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 1. Force set batch_size to the desired number of samples (e.g., 8)
    dataloaders = create_data_loaders(
        data_root=args.data_root,
        batch_size=args.num_samples, 
        num_workers=4,
        patch_size=256, # Use full image or large patches during testing -> use 256 to avoid OOM
        use_augmentation=False # Do NOT use data augmentation during testing!
    )
    test_loader = dataloaders['test'] # Use the test set!

    # 2. Identify and build the model
    arch = config['model']['generator'].get('arch', 'unet')
    print(f"Loading Model Architecture: {arch}")
    
    in_nc = config['model']['generator'].get('input_nc', 3)
    out_nc = config['model']['generator'].get('output_nc', 3)
    
    # Dynamically load the Generator for the corresponding model
    if arch in ['dgr', 'dgr_dtr']:
        full_model = create_dgr_model(config['model']['generator'])
        ckpt = torch.load(args.ckpt, map_location=device)
        full_model.load_state_dict(ckpt['model_state_dict'])
        netG = full_model.generator
    elif arch == 'dual_branch':
        netG = DualBranchGenerator(input_nc=in_nc, output_nc=out_nc) 
        ckpt = torch.load(args.ckpt, map_location=device)
        netG.load_state_dict(ckpt['netG_state_dict'])
    else:
        netG = create_generator(arch, input_nc=in_nc, output_nc=out_nc)
        ckpt = torch.load(args.ckpt, map_location=device)
        netG.load_state_dict(ckpt['netG_state_dict'])

    netG.to(device)
    netG.eval()

    # 3. Grab one batch for inference
    print("Running inference on Test Set...")
    with torch.no_grad():
        batch = next(iter(test_loader))
        input_img = batch['input'].to(device)
        label_img = batch['label'].to(device)

        # Adapt to the [-1, 1] input logic of DTR
        if arch in ['dgr', 'dgr_dtr']:
            model_input = input_img * 2.0 - 1.0
        else:
            model_input = input_img

        # [GPU Memory Fix] Split batch=8 with a for loop, infer 1 image at a time to bypass Attention peak memory usage
        fake_imgs = []
        for i in range(model_input.size(0)):
            single_input = model_input[i:i+1]  # Extract single image, keep [1, C, H, W] dimension
            single_fake = netG(single_input)   # Infer single image to release peak GPU memory
            fake_imgs.append(single_fake)
        
        # Reassemble the 8 generated single images back into a Batch [8, C, H, W]
        fake_img = torch.cat(fake_imgs, dim=0)

    # 4. Stitch and save images
    print("Saving figures...")
    # save input H&E (all channels)
    vutils.save_image(denorm(input_img), os.path.join(args.output_dir, f"{arch}_input_HE.png"), nrow=4)
    # save prediction mIHC
    vutils.save_image(denorm(fake_img), os.path.join(args.output_dir, f"{arch}_pred_mIHC.png"), nrow=4)
    # save ground truth mIHC
    vutils.save_image(denorm(label_img), os.path.join(args.output_dir, f"{arch}_real_mIHC.png"), nrow=4)

    print(f"Done! 8-sample comparison images saved to '{args.output_dir}'")

if __name__ == '__main__':
    main()