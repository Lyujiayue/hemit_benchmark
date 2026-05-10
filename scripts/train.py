"""
Unified Training Script for HEMIT Benchmark

Supports: pix2pix (UNet/ResNet), Dual-Branch (HEMIT), DGR/DTR

Usage:
    # pix2pix UNet
    python scripts/train.py --config configs/pix2pix_unet.yaml --data_root /path/to/HEMIT --exp_name pix2pix_unet

    # Dual-Branch (HEMIT paper)
    python scripts/train.py --config configs/dual_branch.yaml --data_root /path/to/HEMIT --exp_name dual_branch

    # DGR
    python scripts/train.py --config configs/dgr.yaml --data_root /path/to/HEMIT --exp_name dgr
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Type
import argparse
import random
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import create_data_loaders
from models.baselines.pix2pix import create_generator, create_discriminator
from models.baselines.dual_branch import DualBranchGenerator, DualBranchDiscriminator
from models.advanced.dgr import DGRGenerator, DGRDiscriminator, DGRLossCalculator
from utils.metrics import MetricsCalculator, print_metrics_table, AggregateMetrics


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    'pix2pix_unet': {
        'generator': 'unet',
        'discriminator': 'patchgan',
        'model_class': 'pix2pix'
    },
    'pix2pix_resnet': {
        'generator': 'resnet',
        'discriminator': 'patchgan',
        'model_class': 'pix2pix'
    },
    'dual_branch': {
        'generator': 'dual_branch',
        'discriminator': 'dual_branch',
        'model_class': 'dual_branch'
    },
    'dgr': {
        'generator': 'dgr',
        'discriminator': 'dgr',
        'model_class': 'dgr'
    }
}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class HEMITTrainer:
    """Unified trainer for all HEMIT benchmark models."""

    def __init__(
        self,
        config: Dict,
        experiment_name: str,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.exp_name = experiment_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = config.get('model_class', 'pix2pix')

        # Output directories
        self.exp_dir = Path(config.get('exp_dir', 'experiments')) / experiment_name
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.vis_dir = self.exp_dir / 'visualizations'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.exp_dir / 'logs'))

        # Build models
        self._build_models()

        # Loss
        self._build_losses()

        # Optimizers
        self._build_optimizers()

        # Schedulers
        self._build_schedulers()

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.metrics_calc = MetricsCalculator()

        print(f"[HEMITTrainer] Device: {self.device}")
        print(f"[HEMITTrainer] Model class: {self.model_class}")

    def _build_models(self):
        gen_cfg = self.config['model']['generator']
        dis_cfg = self.config['model']['discriminator']

        arch = gen_cfg.get('arch', 'unet')

        # ---- Generator ----
        if arch == 'dual_branch':
            self.netG = DualBranchGenerator(
                input_nc=gen_cfg.get('input_nc', 1),
                output_nc=gen_cfg.get('output_nc', 3),
                ngf=gen_cfg.get('ngf', 64)
            )
        elif arch == 'dgr':
            self.netG = DGRGenerator(
                input_nc=gen_cfg.get('input_nc', 1),
                output_nc=gen_cfg.get('output_nc', 3),
                ngf=gen_cfg.get('ngf', 64)
            )
        else:
            self.netG = create_generator(
                arch=arch,
                input_nc=gen_cfg.get('input_nc', 1),
                output_nc=gen_cfg.get('output_nc', 3),
                ngf=gen_cfg.get('ngf', 64)
            )

        self.netG.to(self.device)

        # ---- Discriminator ----
        dis_input_nc = gen_cfg.get('output_nc', 3) + gen_cfg.get('input_nc', 1)

        if arch == 'dual_branch':
            self.netD = DualBranchDiscriminator(
                input_nc=dis_input_nc,
                ndf=dis_cfg.get('ndf', 64),
                n_layers=dis_cfg.get('n_layers', 3)
            )
        elif arch == 'dgr':
            self.netD = DGRDiscriminator(
                input_nc=dis_input_nc,
                ndf=dis_cfg.get('ndf', 64),
                n_layers=dis_cfg.get('n_layers', 3)
            )
        else:
            self.netD = create_discriminator(
                input_nc=dis_input_nc,
                ndf=dis_cfg.get('ndf', 64),
                n_layers=dis_cfg.get('n_layers', 3)
            )

        self.netD.to(self.device)

        print(f"[HEMITTrainer] Generator: {arch} (params: {self.netG.get_num_parameters() if hasattr(self.netG, 'get_num_parameters') else sum(p.numel() for p in self.netG.parameters()):,})")
        print(f"[HEMITTrainer] Discriminator: PatchGAN")

    def _build_losses(self):
        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()

        if self.model_class == 'dgr':
            self.dgr_loss_calc = DGRLossCalculator(
                lambda_L1=self.config['training'].get('lambda_L1', 100.0),
                lambda_fm=self.config['training'].get('lambda_fm', 1.0)
            )

    def _build_optimizers(self):
        opt_cfg = self.config['optimizer']
        lr = opt_cfg.get('lr', 2e-4)

        self.optimizerG = optim.Adam(
            self.netG.parameters(),
            lr=lr,
            betas=(opt_cfg.get('beta1', 0.5), opt_cfg.get('beta2', 0.999))
        )
        self.optimizerD = optim.Adam(
            self.netD.parameters(),
            lr=lr,
            betas=(opt_cfg.get('beta1', 0.5), opt_cfg.get('beta2', 0.999))
        )

    def _build_schedulers(self):
        sched_cfg = self.config.get('scheduler', {})
        policy = sched_cfg.get('policy', 'step')

        if policy == 'step':
            self.schedG = optim.lr_scheduler.StepLR(
                self.optimizerG,
                step_size=sched_cfg.get('step_size', 50),
                gamma=sched_cfg.get('gamma', 0.5)
            )
            self.schedD = optim.lr_scheduler.StepLR(
                self.optimizerD,
                step_size=sched_cfg.get('step_size', 50),
                gamma=sched_cfg.get('gamma', 0.5)
            )
        else:
            self.schedG = optim.lr_scheduler.StepLR(self.optimizerG, step_size=1000)
            self.schedD = optim.lr_scheduler.StepLR(self.optimizerD, step_size=1000)

    def train_step(self, input_img: torch.Tensor, label_img: torch.Tensor) -> Dict[str, float]:
        # Forward
        fake_img = self.netG(input_img)

        # --- Discriminator update ---
        self.optimizerD.zero_grad()

        # Real
        pred_real = self.netD(label_img, input_img)
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))

        # Fake
        pred_fake = self.netD(fake_img.detach(), input_img)
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.optimizerD.step()

        # --- Generator update ---
        self.optimizerG.zero_grad()

        pred_fake_g = self.netD(fake_img, input_img)
        loss_G_GAN = self.criterionGAN(pred_fake_g, torch.ones_like(pred_fake_g))

        lambda_L1 = self.config['training'].get('lambda_L1', 100.0)
        loss_G_L1 = self.criterionL1(fake_img, label_img) * lambda_L1

        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        self.optimizerG.step()

        return {
            'loss_G': loss_G.item(),
            'loss_G_GAN': loss_G_GAN.item(),
            'loss_G_L1': loss_G_L1.item(),
            'loss_D': loss_D.item(),
            'loss_D_real': loss_D_real.item(),
            'loss_D_fake': loss_D_fake.item()
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict:
        self.netG.eval()

        all_metrics = []
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            input_img = batch['input'].to(self.device)
            label_img = batch['label'].to(self.device)

            fake_img = self.netG(input_img)

            for i in range(fake_img.size(0)):
                real_np = label_img[i].cpu().numpy()
                fake_np = fake_img[i].cpu().numpy()

                if real_np.shape[0] == 3:
                    real_np = np.transpose(real_np, (1, 2, 0))
                    fake_np = np.transpose(fake_np, (1, 2, 0))

                img_metrics = self.metrics_calc.compute_image_metrics(real_np, fake_np, batch['filename'][i])
                all_metrics.append(img_metrics)

        self.netG.train()
        return AggregateMetrics(**self.metrics_calc.aggregate_metrics(all_metrics).__dict__)

    def visualize_batch(self, input_img, fake_img, real_img, step, num_vis=4):
        def norm(t):
            t = t.clone()
            for i in range(t.size(0)):
                ch = t[i]
                mn, mx = ch.min(), ch.max()
                if mx - mn > 1e-6:
                    t[i] = (ch - mn) / (mx - mn)
            return t

        # Input H&E (single channel)
        inp_grid = vutils.make_grid(norm(input_img[:num_vis, :1]), nrow=num_vis)
        fake_grid = vutils.make_grid(norm(fake_img[:num_vis]), nrow=num_vis)
        real_grid = vutils.make_grid(norm(real_img[:num_vis]), nrow=num_vis)

        self.writer.add_image('val/input_HE', inp_grid, step)
        self.writer.add_image('val/fake_mIHC', fake_grid, step)
        self.writer.add_image('val/real_mIHC', real_grid, step)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'config': self.config
        }

        torch.save(ckpt, self.ckpt_dir / f'epoch_{epoch:04d}.pth')
        torch.save(ckpt, self.ckpt_dir / 'latest.pth')

        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')

    def load_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.netG.load_state_dict(ckpt['netG_state_dict'])
        self.netD.load_state_dict(ckpt['netD_state_dict'])
        self.optimizerG.load_state_dict(ckpt['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(ckpt['optimizerD_state_dict'])
        self.current_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, val_freq: int = 5, save_freq: int = 10):
        print(f"\n{'=' * 60}")
        print(f"Training: {self.exp_name} | {num_epochs} epochs")
        print(f"{'=' * 60}")

        best_metric = -float('inf')
        loss_keys = ['loss_G', 'loss_G_GAN', 'loss_G_L1', 'loss_D', 'loss_D_real', 'loss_D_fake']

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            self.netG.train()
            self.netD.train()

            epoch_losses = {k: [] for k in loss_keys}
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

            for batch in pbar:
                input_img = batch['input'].to(self.device)
                label_img = batch['label'].to(self.device)

                losses = self.train_step(input_img, label_img)

                for k in loss_keys:
                    epoch_losses[k].append(losses[k])

                self.global_step += 1

                if self.global_step % 100 == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f'loss/{k}', v, self.global_step)

                pbar.set_postfix({
                    'G': f"{np.mean(epoch_losses['loss_G']):.4f}",
                    'D': f"{np.mean(epoch_losses['loss_D']):.4f}"
                })

            # Epoch summary
            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(f"Epoch {epoch}: G={avg['loss_G']:.4f} | G_GAN={avg['loss_G_GAN']:.4f} "
                  f"| G_L1={avg['loss_G_L1']:.4f} | D={avg['loss_D']:.4f}")

            # LR schedule
            self.schedG.step()
            self.schedD.step()

            # Validation
            if epoch % val_freq == 0:
                print(f"\nValidation at epoch {epoch}...")
                agg = self.validate(val_loader)
                print_metrics_table(agg, self.exp_name)

                for metric_name, ch_dict in [('SSIM', agg.to_dict()['ssim']),
                                              ('Pearson', agg.to_dict()['pearson']),
                                              ('PSNR', agg.to_dict()['psnr'])]:
                    for ch, val in ch_dict.items():
                        self.writer.add_scalar(f'val/{metric_name}/{ch}', val['mean'], epoch)

                is_best = agg.to_dict()['ssim']['Average']['mean'] > best_metric
                if is_best:
                    best_metric = agg.to_dict()['ssim']['Average']['mean']

                self.save_checkpoint(epoch, is_best=is_best)

                # Visualize
                with torch.no_grad():
                    sample = next(iter(val_loader))
                    inp = sample['input'].to(self.device)
                    lab = sample['label'].to(self.device)
                    fak = self.netG(inp)
                    self.visualize_batch(inp, fak, lab, epoch)

            elif epoch % save_freq == 0:
                self.save_checkpoint(epoch)

        print(f"\nTraining complete! Best SSIM: {best_metric:.4f}")
        self.writer.close()
        return best_metric


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    set_seed(args.seed)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs

    # Override data root
    if 'data_root' not in config:
        config['data_root'] = args.data_root

    # Data loaders
    data_cfg = config.get('data', {})
    dataloaders = create_data_loaders(
        data_root=args.data_root,
        batch_size=config['training'].get('batch_size', 2),
        num_workers=data_cfg.get('num_workers', 4),
        patch_size=data_cfg.get('patch_size', None),
        use_augmentation=data_cfg.get('use_augmentation', True)
    )

    # Trainer
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    trainer = HEMITTrainer(config=config, experiment_name=args.exp_name, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    best = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=config['training'].get('num_epochs', 100),
        val_freq=config['training'].get('val_freq', 5),
        save_freq=config['training'].get('save_freq', 10)
    )

    # Save config
    with open(trainer.exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    main()
