"""
Failure Case Miner
Runs through the test set, calculates L1 error for each image, 
and saves the worst N cases (highest error).
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
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
    t = t.clone()
    for i in range(t.size(0)):
        ch = t[i]
        mn, mx = ch.min(), ch.max()
        if mx - mn > 1e-6:
            t[i] = (ch - mn) / (mx - mn)
    return t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="report_figures/failures")
    parser.add_argument('--num_failures', type=int, default=4, help="Number of worst cases to save")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Traverse the test set with Batch Size 1 to accurately locate single images
    dataloaders = create_data_loaders(
        data_root=args.data_root,
        batch_size=1, 
        num_workers=4,
        patch_size=256, # for advanced baselines, to avoid OOM
        use_augmentation=False
    )
    test_loader = dataloaders['test']

    arch = config['model']['generator'].get('arch', 'unet')
    in_nc = config['model']['generator'].get('input_nc', 3)
    out_nc = config['model']['generator'].get('output_nc', 3)
    
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

    criterion = nn.L1Loss(reduction='sum')
    error_list = []

    print("Mining test set for highest error samples...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_img = batch['input'].to(device)
            label_img = batch['label'].to(device)

            if arch in ['dgr', 'dgr_dtr']:
                model_input = input_img * 2.0 - 1.0
            else:
                model_input = input_img

            fake_img = netG(model_input)
            
            # Compatible with DGR: restore to [0,1] for error calculation
            if arch in ['dgr', 'dgr_dtr']:
                fake_img_for_loss = torch.clamp((fake_img + 1.0) / 2.0, 0.0, 1.0)
            else:
                fake_img_for_loss = fake_img

            # Calculate absolute error
            loss = criterion(fake_img_for_loss, label_img).item()
            
            # Store the data
            error_list.append({
                'loss': loss,
                'filename': batch['filename'][0],
                'input': input_img.cpu(),
                'fake': fake_img.cpu(),
                'label': label_img.cpu()
            })

    # Sort by loss from largest to smallest (higher error comes first)
    error_list.sort(key=lambda x: x['loss'], reverse=True)

    # Extract the worst N images
    worst_cases = error_list[:args.num_failures]
    
    print(f"\nFound Top {args.num_failures} Failure Cases:")
    for i, item in enumerate(worst_cases):
        print(f"Rank {i+1} | File: {item['filename']} | L1 Error: {item['loss']:.2f}")

    # Concatenate and save
    inputs = torch.cat([item['input'] for item in worst_cases])
    fakes = torch.cat([item['fake'] for item in worst_cases])
    labels = torch.cat([item['label'] for item in worst_cases])

    vutils.save_image(denorm(inputs), os.path.join(args.output_dir, f"{arch}_worst_input_HE.png"), nrow=args.num_failures)
    vutils.save_image(denorm(fakes), os.path.join(args.output_dir, f"{arch}_worst_pred_mIHC.png"), nrow=args.num_failures)
    vutils.save_image(denorm(labels), os.path.join(args.output_dir, f"{arch}_worst_real_mIHC.png"), nrow=args.num_failures)

    print(f"\nFailure cases saved to {args.output_dir}")

if __name__ == '__main__':
    main()