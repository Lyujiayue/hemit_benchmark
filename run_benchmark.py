"""
One-Command Benchmark Runner

Runs the full HEMIT benchmark pipeline:
1. Data validation
2. Training all methods
3. Evaluation and results

Usage:
    python run_benchmark.py --data_root /path/to/HEMIT --method all
    python run_benchmark.py --data_root /path/to/HEMIT --method pix2pix_unet
    python run_benchmark.py --data_root /path/to/HEMIT --method dgr --download_weights
"""
import os
import sys
from pathlib import Path
from typing import List, Optional
import argparse
import json
import shutil
import subprocess

BASE_DIR = Path(__file__).parent


class BenchmarkRunner:
    """Orchestrates the full HEMIT benchmark pipeline."""

    METHODS = {
        'pix2pix_unet': {
            'config': 'configs/pix2pix_unet.yaml',
            'epochs': 100,
            'description': 'Pix2pix with U-Net generator (classic baseline)'
        },
        'pix2pix_resnet': {
            'config': 'configs/pix2pix_resnet.yaml',
            'epochs': 100,
            'description': 'Pix2pix with ResNet generator (classic baseline)'
        },
        'dual_branch': {
            'config': 'configs/dual_branch.yaml',
            'epochs': 80,
            'description': 'Dual-Branch Pix2pix (HEMIT paper method)'
        },
        'dgr': {
            'config': 'configs/dgr.yaml',
            'epochs': 100,
            'description': 'DGR/DTR misalignment-resistant virtual staining'
        }
    }

    def __init__(self, data_root: str, output_root: str = './results'):
        self.data_root = data_root
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def validate_data(self):
        """Step 1: Validate dataset integrity."""
        print("\n" + "=" * 60)
        print("STEP 1: Data Validation")
        print("=" * 60)

        cmd = [
            sys.executable,
            str(BASE_DIR / 'scripts' / 'prepare_data.py'),
            '--data_path', self.data_root,
            '--output_path', str(self.output_root / 'data_prep'),
            '--num_vis_samples', '8'
        ]

        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0

    def download_weights(self, method: str):
        """Download pretrained weights for a method."""
        if method == 'dgr':
            ckpt_dir = BASE_DIR / 'checkpoints'
            ckpt_dir.mkdir(exist_ok=True)
            dst = ckpt_dir / 'dgr_hemit_pretrained.pth'

            if dst.exists():
                print(f"  DGR weights already exist at {dst}")
                return

            print(f"  Downloading DGR HEMIT pretrained weights...")
            url = "https://github.com/birkhoffkiki/DTR/releases/download/weights/hemit_weight.pth"
            try:
                import urllib.request
                urllib.request.urlretrieve(url, str(dst))
                print(f"  Saved to {dst}")
            except Exception as e:
                print(f"  WARNING: Could not download DGR weights: {e}")
                print(f"  Download manually from: {url}")

        elif method == 'dual_branch':
            ckpt_dir = BASE_DIR / 'checkpoints'
            ckpt_dir.mkdir(exist_ok=True)
            dst = ckpt_dir / 'dual_branch_hemit.pth'

            if dst.exists():
                print(f"  Dual-Branch weights already exist at {dst}")
                return

            print(f"  Downloading Dual-Branch HEMIT pretrained weights...")
            url = "https://drive.google.com/file/d/1HNc-dj2ATN7gdAyOCy-lWe8_YQse2CTd/view?usp=sharing"
            print(f"  Manual download required: {url}")
            print(f"  Save as: {dst}")

    def train_method(self, method: str, epochs: Optional[int] = None, resume: Optional[str] = None):
        """Step 2: Train a single method."""
        method_cfg = self.METHODS.get(method)
        if not method_cfg:
            print(f"Unknown method: {method}")
            return False

        config_path = BASE_DIR / method_cfg['config']
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            return False

        print("\n" + "=" * 60)
        print(f"STEP 2: Training {method}")
        print(f"Description: {method_cfg['description']}")
        print("=" * 60)

        cmd = [
            sys.executable,
            str(BASE_DIR / 'scripts' / 'train.py'),
            '--config', str(config_path),
            '--data_root', self.data_root,
            '--exp_name', method,
            '--seed', '42'
        ]

        if epochs is not None:
            cmd.extend(['--epochs', str(epochs)])
        elif method_cfg.get('epochs'):
            cmd.extend(['--epochs', str(method_cfg['epochs'])])

        if resume:
            cmd.extend(['--resume', resume])

        result = subprocess.run(cmd)
        return result.returncode == 0

    def evaluate_method(self, method: str):
        """Step 3: Evaluate a trained method."""
        print("\n" + "=" * 60)
        print(f"STEP 3: Evaluating {method}")
        print("=" * 60)

        ckpt_path = BASE_DIR / 'experiments' / method / 'checkpoints' / 'best.pth'
        if not ckpt_path.exists():
            print(f"  WARNING: No checkpoint found at {ckpt_path}, skipping evaluation.")
            return None

        config_path = BASE_DIR / self.METHODS[method]['config']
        output_dir = self.output_root / 'eval' / method

        cmd = [
            sys.executable,
            str(BASE_DIR / 'scripts' / 'evaluate.py'),
            '--config', str(config_path),
            '--checkpoint', str(ckpt_path),
            '--data_root', self.data_root,
            '--split', 'test',
            '--output_dir', str(output_dir)
        ]

        result = subprocess.run(cmd, capture_output=False)

        # Parse results
        results_file = output_dir / 'metrics_summary.json'
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
        return None

    def run_full_benchmark(self, methods: List[str], train_epochs: Optional[int] = None,
                           skip_training: bool = False, skip_evaluation: bool = False):
        """Run the full benchmark pipeline."""
        print("\n" + "=" * 70)
        print("HEMIT BENCHMARK PIPELINE")
        print("=" * 70)
        print(f"Data root: {self.data_root}")
        print(f"Output root: {self.output_root}")
        print(f"Methods: {methods}")
        print(f"Train epochs: {train_epochs or 'default'}")
        print(f"Skip training: {skip_training}")
        print("=" * 70)

        # Step 0: Validate data
        self.validate_data()

        # Step 1: Download weights
        for method in methods:
            if method in ('dgr', 'dual_branch'):
                self.download_weights(method)

        # Step 2: Train
        if not skip_training:
            for method in methods:
                self.train_method(method, epochs=train_epochs)

        # Step 3: Evaluate
        if not skip_evaluation:
            all_results = {}
            for method in methods:
                results = self.evaluate_method(method)
                if results:
                    all_results[method] = results

            # Save summary table
            self._save_summary_table(all_results)

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)

    def _save_summary_table(self, results: dict):
        """Save a markdown summary table of all results."""
        lines = [
            "# HEMIT Benchmark Results",
            "",
            "| Method | SSIM (Avg) | Pearson (Avg) | PSNR (Avg) |",
            "|--------|-----------|-------------|-----------|"
        ]

        for method, r in results.items():
            ssim = r.get('ssim', {}).get('Average', {}).get('mean', 'N/A')
            pearson = r.get('pearson', {}).get('Average', {}).get('mean', 'N/A')
            psnr = r.get('psnr', {}).get('Average', {}).get('mean', 'N/A')

            ssim_str = f"{ssim:.4f}" if isinstance(ssim, float) else str(ssim)
            pearson_str = f"{pearson:.4f}" if isinstance(pearson, float) else str(pearson)
            psnr_str = f"{psnr:.2f}" if isinstance(psnr, float) else str(psnr)

            lines.append(f"| {method} | {ssim_str} | {pearson_str} | {psnr_str} |")

            # Per-channel
            lines.append(f"|  ├─ DAPI | {r.get('ssim',{}).get('DAPI',{}).get('mean','N/A'):.4f} | {r.get('pearson',{}).get('DAPI',{}).get('mean','N/A'):.4f} | {r.get('psnr',{}).get('DAPI',{}).get('mean','N/A'):.2f} |")
            lines.append(f"|  ├─ panCK | {r.get('ssim',{}).get('panCK',{}).get('mean','N/A'):.4f} | {r.get('pearson',{}).get('panCK',{}).get('mean','N/A'):.4f} | {r.get('psnr',{}).get('panCK',{}).get('mean','N/A'):.2f} |")
            lines.append(f"|  └─ CD3 | {r.get('ssim',{}).get('CD3',{}).get('mean','N/A'):.4f} | {r.get('pearson',{}).get('CD3',{}).get('mean','N/A'):.4f} | {r.get('psnr',{}).get('CD3',{}).get('mean','N/A'):.2f} |")

        summary_path = self.output_root / 'benchmark_summary.md'
        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nBenchmark summary saved to {summary_path}")
        print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='HEMIT Benchmark Runner')
    parser.add_argument('--data_root', type=str, required=True, help='Path to HEMIT dataset')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'pix2pix_unet', 'pix2pix_resnet', 'dual_branch', 'dgr'])
    parser.add_argument('--output_root', type=str, default='./results')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_evaluation', action='store_true')
    parser.add_argument('--download_weights', action='store_true')

    args = parser.parse_args()

    runner = BenchmarkRunner(args.data_root, args.output_root)

    if args.download_weights:
        if args.method == 'all':
            runner.download_weights('dgr')
            runner.download_weights('dual_branch')
        else:
            runner.download_weights(args.method)
        return

    if args.method == 'all':
        methods = ['pix2pix_unet', 'pix2pix_resnet', 'dual_branch', 'dgr']
    else:
        methods = [args.method]

    runner.run_full_benchmark(
        methods=methods,
        train_epochs=args.epochs,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )


if __name__ == '__main__':
    main()
