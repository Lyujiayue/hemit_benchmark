"""
Exclusive Training Script for DGR/DTR Baseline
Includes full checkpoint resuming and identical output logging.
"""
import os
import sys
from pathlib import Path
from typing import Dict
import argparse
import random
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import create_data_loaders
from models.advanced.dgr_dtr import create_dgr_model
from utils.metrics import MetricsCalculator, print_metrics_table, AggregateMetrics


class DTRTrainer:
    def __init__(self, config: Dict, experiment_name: str, device: torch.device):
        self.config = config
        self.exp_name = experiment_name
        self.device = device

        # Output directories
        self.exp_dir = Path(config.get('exp_dir', 'experiments')) / experiment_name
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.vis_dir = self.exp_dir / 'visualizations'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.exp_dir / 'logs'))

        # Build Model 
        print("[DTRTrainer] Building DGRModel...")
        self.model = create_dgr_model(config['model']['generator'])
        self.model.to(self.device)

        # Build Optimizers 
        opt_cfg = self.config['optimizer']
        lr = opt_cfg.get('lr', 2e-4)
        self.optimizerG = optim.Adam(
            self.model.generator.parameters(),
            lr=lr, betas=(opt_cfg.get('beta1', 0.5), opt_cfg.get('beta2', 0.999))
        )
        self.optimizerD = optim.Adam(
            self.model.discriminator.parameters(),
            lr=lr*0.1, betas=(opt_cfg.get('beta1', 0.5), opt_cfg.get('beta2', 0.999))
        )

        # Build Schedulers
        sched_cfg = self.config.get('scheduler', {})
        step_size = sched_cfg.get('step_size', 50)
        gamma = sched_cfg.get('gamma', 0.5)
        self.schedG = optim.lr_scheduler.StepLR(self.optimizerG, step_size=step_size, gamma=gamma)
        self.schedD = optim.lr_scheduler.StepLR(self.optimizerD, step_size=step_size, gamma=gamma)

        self.current_epoch = 0
        self.global_step = 0
        self.metrics_calc = MetricsCalculator()

    def train_step(self, input_img: torch.Tensor, label_img: torch.Tensor) -> Dict[str, float]:
        outputs = self.model(real_A=input_img, real_B=label_img, training=True)
        # ==========================================
        # Core Fix: Update Generator first, then update Discriminator
        # to avoid in-place operation version mismatch.
        # ==========================================
        
        # --- 1. Backpropagate and update Generator first ---
        self.optimizerG.zero_grad()
        # retain_graph=True ensures the computational graph is kept for loss_D backward pass
        outputs['loss_G'].backward(retain_graph=True)
        self.optimizerG.step()
        # Note: loss_G.backward() accumulates unwanted gradients in D, 
        # but it's safe since we haven't performed optimizerD.step() yet.

        # --- 2. Backpropagate and update Discriminator ---
        # CRITICAL: This zero_grad() clears out the unwanted gradients left by G's backward pass!
        self.optimizerD.zero_grad() 
        outputs['loss_D'].backward()
        self.optimizerD.step()

        return {
            'loss_G': outputs['loss_G'].item(),
            'loss_D': outputs['loss_D'].item(),
            'loss_L1': outputs.get('loss_L1', torch.tensor(0)).item(),
            'loss_perceptual': outputs.get('loss_perceptual', torch.tensor(0)).item(),
            'loss_consistency': outputs.get('loss_consistency', torch.tensor(0)).item()
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> AggregateMetrics:
        self.model.eval()
        all_metrics = []
        
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            input_img = batch['input'].to(self.device)*2.0 - 1.0
            label_img = batch['label'].to(self.device)

            fake_img = self.model.inference(input_img)
            # Convert fake_img from [-1, 1] to [0, 1]
            fake_img = (fake_img + 1.0) / 2.0
            fake_img = torch.clamp(fake_img, 0.0, 1.0)

            for i in range(fake_img.size(0)):
                real_np = label_img[i].cpu().numpy()
                fake_np = fake_img[i].cpu().numpy()

                if real_np.shape[0] == 3:
                    real_np = np.transpose(real_np, (1, 2, 0))
                    fake_np = np.transpose(fake_np, (1, 2, 0))

                img_metrics = self.metrics_calc.compute_image_metrics(real_np, fake_np, batch['filename'][i])
                all_metrics.append(img_metrics)

        self.model.train()
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

        inp_grid = vutils.make_grid(norm(input_img[:num_vis, :1]), nrow=num_vis)
        fake_grid = vutils.make_grid(norm(fake_img[:num_vis]), nrow=num_vis)
        real_grid = vutils.make_grid(norm(real_img[:num_vis]), nrow=num_vis)

        self.writer.add_image('val/input_HE', inp_grid, step)
        self.writer.add_image('val/fake_mIHC', fake_grid, step)
        self.writer.add_image('val/real_mIHC', real_grid, step)

        
        vutils.save_image(inp_grid, self.vis_dir / f'epoch_{step:04d}_input.png')
        vutils.save_image(fake_grid, self.vis_dir / f'epoch_{step:04d}_fake.png')
        vutils.save_image(real_grid, self.vis_dir / f'epoch_{step:04d}_real.png')

    def load_checkpoint(self, ckpt_path: str):
        """Fix: Core logic for resuming training from checkpoint"""
        print(f"[DTRTrainer] Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizerG.load_state_dict(ckpt['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(ckpt['optimizerD_state_dict'])
        self.current_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        print(f"[DTRTrainer] Successfully resumed. Will start from epoch {self.current_epoch + 1}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'config': self.config
        }
        torch.save(ckpt, self.ckpt_dir / f'epoch_{epoch:04d}.pth')
        torch.save(ckpt, self.ckpt_dir / 'latest.pth')
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, val_freq: int, save_freq: int):
        print(f"\n{'=' * 60}")
        print(f"Training DTR: {self.exp_name} | {num_epochs} epochs")
        print(f"{'=' * 60}")

        best_metric = -float('inf')
        loss_keys = ['loss_G', 'loss_D', 'loss_L1', 'loss_perceptual', 'loss_consistency']

        for epoch in range(self.current_epoch + 1, num_epochs + 1):
            self.current_epoch = epoch
            self.model.train()

            epoch_losses = {k: [] for k in loss_keys}
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

            for batch in pbar:
                input_img = batch['input'].to(self.device)*2.0 - 1.0
                label_img = batch['label'].to(self.device)*2.0 - 1.0

                losses = self.train_step(input_img, label_img)

                for k in loss_keys:
                    epoch_losses[k].append(losses[k])

                self.global_step += 1

                if self.global_step % 100 == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f'loss/{k}', v, self.global_step)

                pbar.set_postfix({'G': f"{losses['loss_G']:.3f}", 'D': f"{losses['loss_D']:.3f}"})

            # Fix: Add terminal summary output at the end of epoch, keep consistent with train.py
            avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(f"Epoch {epoch}: G={avg['loss_G']:.4f} | D={avg['loss_D']:.4f} "
                  f"| L1={avg['loss_L1']:.4f} | Perceptual={avg['loss_perceptual']:.4f}")

            # LR schedule
            self.schedG.step()
            self.schedD.step()

            # Validation
            if epoch % val_freq == 0:
                agg = self.validate(val_loader)
                print_metrics_table(agg, self.exp_name)
                
                ssim_avg = agg.to_dict()['ssim']['Average']['mean']
                is_best = ssim_avg > best_metric
                if is_best:
                    best_metric = ssim_avg
                
                self.save_checkpoint(epoch, is_best=is_best)

                with torch.no_grad():
                    sample = next(iter(val_loader))
                    inp = sample['input'].to(self.device)
                    lab = sample['label'].to(self.device)
                    fak = self.model.inference(inp)
                    self.visualize_batch(inp, fak, lab, epoch)
            elif epoch % save_freq == 0:
                self.save_checkpoint(epoch)

        print(f"\nTraining complete! Best SSIM: {best_metric:.4f}")
        self.writer.close()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)  # Fix: Add checkpoint resuming parameter
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataloaders = create_data_loaders(
        data_root=args.data_root,
        batch_size=config['training'].get('batch_size', 2),
        num_workers=config.get('data', {}).get('num_workers', 4),
        patch_size=config.get('data', {}).get('patch_size', 512)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = DTRTrainer(config=config, experiment_name=args.exp_name, device=device)

    # Fix: Detect and trigger checkpoint loading
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=config['training'].get('num_epochs', 100),
        val_freq=config['training'].get('val_freq', 5),
        save_freq=config['training'].get('save_freq', 10)
    )

if __name__ == '__main__':
    main()