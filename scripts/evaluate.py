"""
Inference and Evaluation Script for HEMIT Benchmark

Generates predictions and computes SSIM, Pearson R, PSNR metrics.

Usage:
    python scripts/evaluate.py --config configs/pix2pix.yaml --checkpoint ckpt.pth --data_root /path/to/HEMIT
    python scripts/inference.py --model pix2pix --checkpoint ckpt.pth --input_dir /path/to/test/input --output_dir ./outputs
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import argparse

import torch
import numpy as np
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baselines.pix2pix import create_generator
from utils.metrics import MetricsCalculator, HEMITEvaluator, print_metrics_table


def load_checkpoint(ckpt_path: str, device: torch.device) -> Dict:
    """Load model checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt


def denormalize(img: np.ndarray) -> np.ndarray:
    """Denormalize from [-1, 1] to [0, 1] or [0, 65535]."""
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    return img


def infer_batch(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """Run inference on a batch."""
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        fake_img = model(input_tensor)

        # Denormalize
        fake_img = denormalize(fake_img.cpu().numpy())

        # Convert CHW to HWC
        if fake_img.shape[1] == 3:
            fake_img = np.transpose(fake_img, (0, 2, 3, 1))

    return fake_img


def run_inference(
    model: torch.nn.Module,
    data_loader,
    output_dir: str,
    device: torch.device,
    save_images: bool = True
):
    """Run inference on a dataset and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    all_metrics = []
    filenames = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference"):
            input_tensor = batch['input'].to(device)
            label_tensor = batch['label'].to(device)
            batch_filenames = batch['filename']

            fake_imgs = infer_batch(model, input_tensor, device)

            # Convert labels to numpy
            label_np = label_tensor.cpu().numpy()
            if label_np.shape[1] == 3:
                label_np = np.transpose(label_np, (0, 2, 3, 1))
            label_np = (label_np + 1) / 2
            label_np = np.clip(label_np, 0, 1)

            # Compute metrics
            for i in range(fake_imgs.shape[0]):
                fake = fake_imgs[i]
                real = label_np[i]
                fname = batch_filenames[i]

                metrics = MetricsCalculator().compute_image_metrics(
                    real, fake, fname
                )
                all_metrics.append(metrics)
                filenames.append(fname)

                if save_images:
                    # Save fake image as TIF
                    fake_tif_path = output_dir / f"fake_B_{Path(fname).stem}.tif"
                    tifffile.imwrite(
                        fake_tif_path,
                        (fake * 255).astype(np.uint8)
                    )

                    # Save real image
                    real_tif_path = output_dir / f"real_B_{Path(fname).stem}.tif"
                    tifffile.imwrite(
                        real_tif_path,
                        (real * 255).astype(np.uint8)
                    )

    return all_metrics, filenames


def visualize_comparisons(
    real_imgs: np.ndarray,
    fake_imgs: np.ndarray,
    input_imgs: np.ndarray,
    filenames: list,
    output_dir: str,
    num_samples: int = 8
):
    """Generate side-by-side comparison visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(num_samples, len(filenames))
    CHANNEL_NAMES = ['DAPI', 'panCK', 'CD3']

    # Separate top performers and worst performers by SSIM
    metrics_calc = MetricsCalculator()
    all_ssim = []
    for i in range(len(filenames)):
        m = metrics_calc.compute_image_metrics(real_imgs[i], fake_imgs[i], filenames[i])
        all_ssim.append(m.average_ssim)

    sorted_indices = np.argsort(all_ssim)

    # Best and worst
    best_indices = sorted_indices[-n // 2:][::-1]
    worst_indices = sorted_indices[:n // 2]

    for group_name, indices in [('best', best_indices), ('worst', worst_indices)]:
        n_cols = 5  # Input + 3 channels + composite
        fig = plt.figure(figsize=(5 * n_cols, 4 * len(indices)))
        gs = gridspec.GridSpec(len(indices), n_cols)

        for row, idx in enumerate(indices):
            fname = filenames[idx]
            inp = input_imgs[idx]
            fake = fake_imgs[idx]
            real = real_imgs[idx]

            # H&E input
            ax0 = fig.add_subplot(gs[row, 0])
            if len(inp.shape) == 3 and inp.shape[2] in [1, 3]:
                ax0.imshow(np.clip(inp[:, :, :3] if inp.shape[2] == 3 else inp[:, :, 0], 0, 1))
            else:
                ax0.imshow(inp[:, :, 0] if len(inp.shape) == 3 else inp, cmap='gray')
            ax0.set_title(f'{fname}\nH&E Input', fontsize=8)
            ax0.axis('off')

            # Per-channel
            for c, ch in enumerate(CHANNEL_NAMES):
                ax = fig.add_subplot(gs[row, c + 1])

                fake_ch = (fake[:, :, c] - fake[:, :, c].min()) / (fake[:, :, c].max() - fake[:, :, c].min() + 1e-8)
                real_ch = (real[:, :, c] - real[:, :, c].min()) / (real[:, :, c].min() + 1e-8)
                diff = np.abs(fake_ch - real_ch)

                # Overlay: fake=magenta, real=green
                composite = np.stack([real_ch, fake_ch, diff * 3], axis=-1)
                ax.imshow(np.clip(composite, 0, 1))
                m = metrics_calc.compute_image_metrics(real, fake, fname)
                ch_ssim = [m.dapi.ssim, m.panck.ssim, m.cd3.ssim][c]
                ax.set_title(f'{ch}\nSSIM={ch_ssim:.3f}', fontsize=8)
                ax.axis('off')

            # Composite
            ax_comp = fig.add_subplot(gs[row, 4])
            fake_n = np.zeros_like(fake)
            real_n = np.zeros_like(real)
            for c in range(3):
                fake_n[:, :, c] = (fake[:, :, c] - fake[:, :, c].min()) / (fake[:, :, c].max() - fake[:, :, c].min() + 1e-8)
                real_n[:, :, c] = (real[:, :, c] - real[:, :, c].min()) / (real[:, :, c].max() - real[:, :, c].min() + 1e-8)

            both = np.stack([real_n[:, :, 0], (real_n[:, :, 1] + fake_n[:, :, 1]) / 2, fake_n[:, :, 2]], axis=-1)
            ax_comp.imshow(np.clip(both, 0, 1))
            ax_comp.set_title(f'Comp\nR/G/F', fontsize=8)
            ax_comp.axis('off')

        plt.suptitle(f'{group_name.capitalize()} Predictions (SSIM={all_ssim[idx]:.3f})', fontsize=12)
        plt.tight_layout()
        out_path = output_dir / f'comparison_{group_name}.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved {group_name} comparisons to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--data_root', type=str, required=True, help='Path to HEMIT dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build model
    gen_cfg = config['model']['generator']
    netG = create_generator(
        arch=gen_cfg.get('arch', 'unet'),
        input_nc=gen_cfg.get('input_nc', 1),
        output_nc=gen_cfg.get('output_nc', 3),
        ngf=gen_cfg.get('ngf', 64)
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    netG.load_state_dict(ckpt['netG_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Data loader
    from data.dataset import HEMITDataset, DataLoader

    dataset = HEMITDataset(
        data_root=args.data_root,
        split=args.split,
        load_mode='lazy'
    )
    loader = DataLoader(
        dataset,
        batch_size=config['training'].get('batch_size', 4),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4)
    )

    # Run inference
    output_dir = Path(args.output_dir) / Path(args.checkpoint).stem / args.split
    all_metrics, filenames = run_inference(netG, loader, str(output_dir), device)

    # Aggregate metrics
    metrics_calc = MetricsCalculator()
    aggregate = metrics_calc.aggregate_metrics(all_metrics)

    print_metrics_table(aggregate, f"{gen_cfg.get('arch', 'pix2pix')}")

    # Save CSV
    csv_path = output_dir / 'metrics.csv'
    from utils.metrics import HEMITEvaluator
    evaluator = HEMITEvaluator(str(output_dir))
    evaluator._save_to_csv(all_metrics, str(csv_path))
    print(f"Metrics saved to {csv_path}")

    # Save JSON
    import json
    json_path = output_dir / 'metrics_summary.json'
    with open(json_path, 'w') as f:
        json.dump(aggregate.to_dict(), f, indent=2)
    print(f"Summary saved to {json_path}")


if __name__ == '__main__':
    main()
