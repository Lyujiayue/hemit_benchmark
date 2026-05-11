"""
Pix2pix Training Module for HEMIT Benchmark

Implements the full pix2pix training loop with:
- GAN loss (BCE) + L1 loss
- Discriminator update with gradient clipping
- Periodic validation and checkpointing
- TensorBoard logging

Reference: Isola et al., CVPR 2017.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
import yaml
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines.pix2pix import (
    create_generator, create_discriminator
)
from data.dataset import create_data_loaders
from utils.metrics import MetricsCalculator, print_metrics_table


class Pix2pixTrainer:
    """Trainer for pix2pix models."""

    def __init__(
        self,
        config: Dict,
        experiment_name: str = "pix2pix_hemit",
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.exp_name = experiment_name

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

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

        # Loss functions
        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()

        # Optimizers
        self._build_optimizers()

        # Schedulers
        self._build_schedulers()

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Metrics
        self.metrics_calc = MetricsCalculator()

        print(f"[Pix2pixTrainer] Device: {self.device}")
        print(f"[Pix2pixTrainer] Experiment: {experiment_name}")

    def _build_models(self):
        """Initialize generator and discriminator."""
        gen_cfg = self.config['model']['generator']
        dis_cfg = self.config['model']['discriminator']

        self.netG = create_generator(
            arch=gen_cfg.get('arch', 'unet'),
            input_nc=gen_cfg.get('input_nc', 1),
            output_nc=gen_cfg.get('output_nc', 3),
            ngf=gen_cfg.get('ngf', 64)
        ).to(self.device)

        # Discriminator input: generated image (3ch) + conditioning input (1ch) = 4ch
        dis_input_nc = gen_cfg.get('output_nc', 3) + gen_cfg.get('input_nc', 1)

        self.netD = create_discriminator(
            input_nc=dis_input_nc,
            ndf=dis_cfg.get('ndf', 64),
            n_layers=dis_cfg.get('n_layers', 3),
            patchgan=dis_cfg.get('patchgan', True)
        ).to(self.device)

        print(f"[Pix2pixTrainer] Generator: {gen_cfg.get('arch', 'unet')}")
        print(f"[Pix2pixTrainer] Discriminator: PatchGAN ({dis_cfg.get('n_layers', 3)} layers)")

    def _build_optimizers(self):
        """Create optimizers."""
        opt_cfg = self.config['optimizer']

        self.optimizerG = optim.Adam(
            self.netG.parameters(),
            lr=opt_cfg.get('lr', 2e-4),
            betas=(opt_cfg.get('beta1', 0.5), opt_cfg.get('beta2', 0.999))
        )

        self.optimizerD = optim.Adam(
            self.netD.parameters(),
            lr=opt_cfg.get('lr', 2e-4),
            betas=(opt_cfg.get('beta1', 0.5), opt_cfg.get('beta2', 0.999))
        )

    def _build_schedulers(self):
        """Create learning rate schedulers."""
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
        elif policy == 'linear':
            self.schedG = optim.lr_scheduler.LinearLR(
                self.optimizerG,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=sched_cfg.get('total_iters', 100)
            )
            self.schedD = optim.lr_scheduler.LinearLR(
                self.optimizerD,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=sched_cfg.get('total_iters', 100)
            )
        else:
            self.schedG = optim.lr_scheduler.StepLR(self.optimizerG, step_size=1000)
            self.schedD = optim.lr_scheduler.StepLR(self.optimizerD, step_size=1000)

    def set_input(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess a training batch."""
        input_img = batch['input'].to(self.device)
        label_img = batch['label'].to(self.device)
        return input_img, label_img

    def forward_G(self, input_img: torch.Tensor, label_img: torch.Tensor) -> Dict:
        """Forward pass through generator."""
        fake_img = self.netG(input_img)
        return {'fake': fake_img, 'real': label_img, 'input': input_img}

    def backward_D(self, input_img: torch.Tensor, fake_img: torch.Tensor, real_img: torch.Tensor):
        """Discriminator backward pass."""
        # Real pair
        pred_real = self.netD(real_img, input_img)
        label_real = torch.ones_like(pred_real)
        loss_D_real = self.criterionGAN(pred_real, label_real)

        # Fake pair
        pred_fake = self.netD(fake_img.detach(), input_img)
        label_fake = torch.zeros_like(pred_fake)
        loss_D_fake = self.criterionGAN(pred_fake, label_fake)

        # Combined
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        loss_D.backward()
        return {'loss_D': loss_D, 'loss_D_real': loss_D_real, 'loss_D_fake': loss_D_fake}

    def backward_G(
        self,
        input_img: torch.Tensor,
        fake_img: torch.Tensor,
        real_img: torch.Tensor
    ) -> Dict:
        """Generator backward pass."""
        # GAN loss
        pred_fake = self.netD(fake_img, input_img)
        loss_G_GAN = self.criterionGAN(pred_fake, torch.ones_like(pred_fake))

        # L1 loss
        lambda_L1 = self.config['training'].get('lambda_L1', 100)
        loss_G_L1 = self.criterionL1(fake_img, real_img) * lambda_L1

        # Total generator loss
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()

        return {
            'loss_G': loss_G,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1
        }

    def train_step(
        self,
        input_img: torch.Tensor,
        label_img: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step."""
        # Forward
        G_outputs = self.forward_G(input_img, label_img)
        fake_img = G_outputs['fake']

        # Update Discriminator
        self.optimizerD.zero_grad()
        D_losses = self.backward_D(input_img, fake_img, label_img)
        self.optimizerD.step()

        # Update Generator
        self.optimizerG.zero_grad()
        G_losses = self.backward_G(input_img, fake_img, label_img)
        self.optimizerG.step()

        losses = {}
        for k, v in {**D_losses, **G_losses}.items():
            losses[k] = float(v.item())

        return losses

    def validate(self, val_loader: DataLoader) -> Dict:
        """Run validation and compute metrics."""
        self.netG.eval()

        all_metrics = []
        val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)

        with torch.no_grad():
            for batch in val_pbar:
                input_img = batch['input'].to(self.device)
                label_img = batch['label'].to(self.device)

                fake_img = self.netG(input_img)

                # Compute metrics
                for i in range(fake_img.size(0)):
                    real_np = label_img[i].cpu().numpy()
                    fake_np = fake_img[i].cpu().numpy()

                    # Convert CHW to HWC
                    if real_np.shape[0] == 3:
                        real_np = np.transpose(real_np, (1, 2, 0))
                        fake_np = np.transpose(fake_np, (1, 2, 0))

                    img_metrics = self.metrics_calc.compute_image_metrics(
                        real_np, fake_np, batch['filename'][i]
                    )
                    all_metrics.append(img_metrics)

                val_pbar.set_postfix({'samples': len(all_metrics)})

        # Aggregate
        aggregate = self.metrics_calc.aggregate_metrics(all_metrics)

        self.netG.train()
        return aggregate.__dict__ if hasattr(aggregate, '__dict__') else aggregate

    def visualize_batch(
        self,
        input_img: torch.Tensor,
        fake_img: torch.Tensor,
        real_img: torch.Tensor,
        step: int,
        num_vis: int = 4
    ):
        """Save visualization of a batch."""
        import numpy as np

        input_vis = input_img[:num_vis].cpu()
        fake_vis = fake_img[:num_vis].cpu()
        real_vis = real_img[:num_vis].cpu()

        # Normalize to [0, 1]
        def norm(t):
            t_min, t_max = t.min(), t.max()
            if t_max - t_min > 1e-6:
                return (t - t_min) / (t_max - t_min)
            return t

        # H&E input: use first channel
        input_grid = vutils.make_grid(norm(input_vis), nrow=num_vis, normalize=True)
        fake_grid = vutils.make_grid(norm(fake_vis), nrow=num_vis, normalize=True)
        real_grid = vutils.make_grid(norm(real_vis), nrow=num_vis, normalize=True)

        self.writer.add_image('Val/Input_HE', input_grid, step)
        self.writer.add_image('Val/Fake_mIHC', fake_grid, step)
        self.writer.add_image('Val/Real_mIHC', real_grid, step)

        # Side-by-side comparison
        comp = torch.cat([norm(input_vis), norm(fake_vis), norm(real_vis)], dim=3)
        comp_grid = vutils.make_grid(comp, nrow=1, normalize=True)
        self.writer.add_image('Val/Comparison_H_E_Fake_Real', comp_grid, step)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'config': self.config
        }

        ckpt_path = self.ckpt_dir / f'epoch_{epoch:04d}.pth'
        torch.save(ckpt, ckpt_path)

        latest_path = self.ckpt_dir / 'latest.pth'
        torch.save(ckpt, latest_path)

        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save(ckpt, best_path)
            print(f"  [Saved] Best checkpoint: {best_path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        val_freq: int = 5,
        save_freq: int = 10
    ):
        """Main training loop."""
        print(f"\n{'=' * 60}")
        print(f"Starting training: {num_epochs} epochs")
        print(f"{'=' * 60}")

        best_metric = -float('inf')

        for epoch in range(self.current_epoch + 1, num_epochs + 1):
            self.current_epoch = epoch
            self.netG.train()
            self.netD.train()

            epoch_losses = {k: [] for k in [
                'loss_G', 'loss_G_GAN', 'loss_G_L1', 'loss_D', 'loss_D_real', 'loss_D_fake'
            ]}

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                input_img, label_img = self.set_input(batch)
                losses = self.train_step(input_img, label_img)

                for k, v in losses.items():
                    if k in epoch_losses:
                        epoch_losses[k].append(v)

                # Logging
                if self.global_step % 100 == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f'Loss/{k}', v, self.global_step)

                self.global_step += 1

                pbar.set_postfix({
                    'loss_G': f"{np.mean(epoch_losses['loss_G']):.4f}",
                    'loss_D': f"{np.mean(epoch_losses['loss_D']):.4f}"
                })

            # Epoch summary
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(f"\nEpoch {epoch}: G={avg_losses['loss_G']:.4f}, "
                  f"G_GAN={avg_losses['loss_G_GAN']:.4f}, G_L1={avg_losses['loss_G_L1']:.4f}, "
                  f"D={avg_losses['loss_D']:.4f}")

            # Step schedulers
            self.schedG.step()
            self.schedD.step()

            # Validation
            if epoch % val_freq == 0:
                print(f"\nRunning validation at epoch {epoch}...")
                val_metrics = self.validate(val_loader)

                # Print metrics table
                from utils.metrics import AggregateMetrics
                agg = AggregateMetrics(**val_metrics)
                print_metrics_table(agg, f"pix2pix-{self.config['model']['generator'].get('arch', 'unet')}")

                # Log to tensorboard
                for metric_name, channel_dict in [('SSIM', agg.to_dict()['ssim']),
                                                    ('Pearson', agg.to_dict()['pearson']),
                                                    ('PSNR', agg.to_dict()['psnr'])]:
                    for ch, val in channel_dict.items():
                        self.writer.add_scalar(f'Val/{metric_name}/{ch}', val['mean'], epoch)

                # Check best
                current_metric = agg.to_dict()['ssim']['Average']['mean']
                is_best = current_metric > best_metric
                if is_best:
                    best_metric = current_metric

                self.save_checkpoint(epoch, is_best=is_best)

                # Visualize
                with torch.no_grad():
                    sample_batch = next(iter(val_loader))
                    input_vis, label_vis = self.set_input(sample_batch)
                    fake_vis = self.netG(input_vis)
                    self.visualize_batch(input_vis, fake_vis, label_vis, epoch)

            elif epoch % save_freq == 0:
                self.save_checkpoint(epoch)

        print(f"\nTraining complete! Best SSIM: {best_metric:.4f}")
        self.writer.close()

    def load_checkpoint(self, ckpt_path: str):
        """Load a checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.netG.load_state_dict(ckpt['netG_state_dict'])
        self.netD.load_state_dict(ckpt['netD_state_dict'])
        self.optimizerG.load_state_dict(ckpt['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(ckpt['optimizerD_state_dict'])
        self.current_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data_root', type=str, required=True, help='Path to HEMIT dataset')
    parser.add_argument('--exp_name', type=str, default='pix2pix_unet')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume from')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    args = parser.parse_args()

    # Seed
    set_seed(args.seed)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs

    # Data loaders
    data_cfg = config.get('data', {})
    dataloaders = create_data_loaders(
        data_root=args.data_root,
        batch_size=config['training'].get('batch_size', 4),
        num_workers=data_cfg.get('num_workers', 4),
        patch_size=data_cfg.get('patch_size', None),
        use_augmentation=data_cfg.get('use_augmentation', True)
    )

    # Trainer
    trainer = Pix2pixTrainer(
        config=config,
        experiment_name=args.exp_name,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=config['training'].get('num_epochs', 100),
        val_freq=config['training'].get('val_freq', 5),
        save_freq=config['training'].get('save_freq', 10)
    )

    # Save final config
    with open(trainer.exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    main()
