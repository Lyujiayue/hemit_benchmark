"""
Dedicated Training Script for MIPHEI-ViT Benchmark (With Resume Support)

Usage:
    # 首次训练
    python scripts/train_vit.py --config configs/miphei_vit.yaml --data_root /root/autodl-tmp --exp_name miphei_vit_baseline
    
    # 断点续训 (假设上次中断在 epoch 5，想接着跑)
    python scripts/train_vit.py --config configs/miphei_vit.yaml --data_root /root/autodl-tmp --exp_name miphei_vit_baseline --resume experiments/miphei_vit_baseline/checkpoints/latest.pth
"""
import os
import sys
from pathlib import Path
from typing import Dict
import argparse
import random

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
from models.advanced.miphei_vit import create_miphei_vit_model
from utils.metrics import MetricsCalculator, print_metrics_table, AggregateMetrics

class ViTTrainer:
    """Specialized trainer for MIPHEI-ViT baseline."""
    def __init__(self, config: Dict, experiment_name: str, device: torch.device):
        self.config = config
        self.exp_name = experiment_name
        self.device = device
        
        # Directories
        self.exp_dir = Path(config.get('exp_dir', 'experiments')) / experiment_name
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.vis_dir = self.exp_dir / 'visualizations'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.exp_dir / 'logs'))

        # Build Models
        self.netG, self.netD = create_miphei_vit_model(config)
        self.netG.to(self.device)
        self.netD.to(self.device)
        print(f"[ViTTrainer] Device: {self.device}")
        print(f"[ViTTrainer] ViT Generator Params: {sum(p.numel() for p in self.netG.parameters()):,}")

        # Losses
        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()
        
        # Optimizers & Schedulers
        opt_cfg = self.config['optimizer']
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt_cfg.get('lr', 1e-4), betas=(opt_cfg.get('beta1', 0.5), opt_cfg.get('beta2', 0.999)))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt_cfg.get('lr', 1e-4), betas=(opt_cfg.get('beta1', 0.5), opt_cfg.get('beta2', 0.999)))
        
        sched_cfg = self.config.get('scheduler', {})
        self.schedG = optim.lr_scheduler.StepLR(self.optimizerG, step_size=sched_cfg.get('step_size', 50), gamma=sched_cfg.get('gamma', 0.5))
        self.schedD = optim.lr_scheduler.StepLR(self.optimizerD, step_size=sched_cfg.get('step_size', 50), gamma=sched_cfg.get('gamma', 0.5))

        self.current_epoch = 0
        self.global_step = 0
        self.metrics_calc = MetricsCalculator()

    def train_step(self, input_img: torch.Tensor, label_img: torch.Tensor) -> Dict[str, float]:
        fake_img = self.netG(input_img)

        # Update D
        self.optimizerD.zero_grad()
        pred_real = self.netD(label_img, input_img)
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        pred_fake = self.netD(fake_img.detach(), input_img)
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)
        self.optimizerD.step()

        # Update G
        self.optimizerG.zero_grad()
        pred_fake_g = self.netD(fake_img, input_img)
        loss_G_GAN = self.criterionGAN(pred_fake_g, torch.ones_like(pred_fake_g)) * self.config['model'].get('lambda_GAN', 1.0)
        loss_G_L1 = self.criterionL1(fake_img, label_img) * self.config['model'].get('lambda_L1', 100.0)
        
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
        self.optimizerG.step()

        return {
            'loss_G': loss_G.item(), 'loss_G_GAN': loss_G_GAN.item(), 'loss_G_L1': loss_G_L1.item(),
            'loss_D': loss_D.item()
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> AggregateMetrics:
        self.netG.eval()
        all_metrics = []
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            input_img = batch['input'].to(self.device)
            label_img = batch['label'].to(self.device)
            
            fake_img = self.netG(input_img)
            
            if torch.isnan(fake_img).any():
                print(f"\n[警告] 发现 NaN 输出！跳过当前 batch 的评测。")
                continue

            for i in range(fake_img.size(0)):
                real_np = label_img[i].cpu().numpy()
                fake_np = fake_img[i].cpu().numpy()
                if real_np.shape[0] == 3:
                    real_np = np.transpose(real_np, (1, 2, 0))
                    fake_np = np.transpose(fake_np, (1, 2, 0))
                img_metrics = self.metrics_calc.compute_image_metrics(real_np, fake_np, batch['filename'][i])
                all_metrics.append(img_metrics)
        self.netG.train()
        return self.metrics_calc.aggregate_metrics(all_metrics)

    def visualize_batch(self, input_img, fake_img, real_img, step, num_vis=4):
        def norm(t):
            t = t.clone()
            for i in range(t.size(0)):
                ch = t[i]
                mn, mx = ch.min(), ch.max()
                if mx - mn > 1e-6: t[i] = (ch - mn) / (mx - mn)
            return t
        
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
        }
        torch.save(ckpt, self.ckpt_dir / 'latest.pth')
        if is_best: 
            torch.save(ckpt, self.ckpt_dir / 'best.pth')

    def load_checkpoint(self, ckpt_path: str):
        """恢复断点状态"""
        print(f"Loading checkpoint from {ckpt_path}...")
        # 加上 weights_only=False 消除 PyTorch 安全警告
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.netG.load_state_dict(ckpt['netG_state_dict'])
        self.netD.load_state_dict(ckpt['netD_state_dict'])
        self.optimizerG.load_state_dict(ckpt['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(ckpt['optimizerD_state_dict'])
        self.current_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        print(f"Successfully loaded! Resuming training from epoch {self.current_epoch + 1}...")

    def train(self, dataloaders: Dict):
        train_loader, val_loader = dataloaders['train'], dataloaders['val']
        num_epochs = self.config['training'].get('num_epochs', 100)
        val_freq = self.config['training'].get('val_freq', 5)
        
        best_metric = -float('inf')
        
        # 从断点的下一个 epoch 开始
        start_epoch = self.current_epoch + 1
        
        print(f"\n{'=' * 60}\nStarting ViT Training: {self.exp_name} | Target: {num_epochs} epochs\n{'=' * 60}")

        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            self.netG.train()
            self.netD.train()
            
            epoch_losses = {'loss_G': [], 'loss_D': [], 'loss_G_L1': []}
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
            
            for batch in pbar:
                losses = self.train_step(batch['input'].to(self.device), batch['label'].to(self.device))
                for k in epoch_losses: epoch_losses[k].append(losses[k])
                self.global_step += 1
                
                if self.global_step % 50 == 0:
                    for k, v in losses.items(): self.writer.add_scalar(f'loss/{k}', v, self.global_step)
                pbar.set_postfix({'G': f"{np.mean(epoch_losses['loss_G']):.4f}", 'D': f"{np.mean(epoch_losses['loss_D']):.4f}"})

            # 更新学习率 (需要额外逻辑：如果调度器包含状态，通常也需要 load_state_dict，但你的配置里调度器比较简单，这里直接 step 也可以)
            self.schedG.step()
            self.schedD.step()

            if epoch % val_freq == 0:
                agg = self.validate(val_loader)
                print_metrics_table(agg, self.exp_name)
                current_ssim = agg.to_dict()['ssim']['Average']['mean']
                
                self.writer.add_scalar('val/SSIM_Avg', current_ssim, epoch)
                self.writer.add_scalar('val/PSNR_Avg', agg.to_dict()['psnr']['Average']['mean'], epoch)
                
                is_best = current_ssim > best_metric
                if is_best: best_metric = current_ssim
                self.save_checkpoint(epoch, is_best=is_best)

                with torch.no_grad():
                    sample = next(iter(val_loader))
                    self.visualize_batch(sample['input'].to(self.device), self.netG(sample['input'].to(self.device)), sample['label'].to(self.device), epoch)

        self.writer.close()
        return best_metric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    # 增加 resume 参数
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training (e.g., latest.pth)')
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataloaders = create_data_loaders(
        data_root=args.data_root,
        batch_size=config['training'].get('batch_size', 2),
        num_workers=config['data'].get('num_workers', 4),
        patch_size=config['data'].get('patch_size', 256),
        use_augmentation=config['data'].get('use_augmentation', True)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ViTTrainer(config=config, experiment_name=args.exp_name, device=device)
    
    # 核心续传逻辑：如果有 resume 参数，先加载模型状态
    if args.resume:
        if os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
        else:
            print(f"Warning: Checkpoint {args.resume} not found. Starting from scratch.")

    with open(trainer.exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    trainer.train(dataloaders)

if __name__ == '__main__':
    main()