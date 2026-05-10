"""
Data Preparation and Validation Script for HEMIT Dataset

This script:
1. Validates the HEMIT dataset structure
2. Checks data integrity (file counts, sizes, channels)
3. Generates exploration visualizations
4. Documents the train/val/test splits
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json
import random

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import HEMITDataValidator


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare and validate HEMIT dataset')
    parser.add_argument('--data_root', type=str, required=True, help='Path to HEMIT dataset root')
    parser.add_argument('--output_dir', type=str, default='./data_exploration', help='Output directory')
    parser.add_argument('--num_visualizations', type=int, default=8, help='Number of visualization samples')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def validate_dataset(data_root: str, output_dir: Path) -> Dict:
    """Validate the dataset and generate report"""
    print("=" * 60)
    print("HEMIT Dataset Validation")
    print("=" * 60)

    validator = HEMITDataValidator(data_root)
    reports = validator.validate_all_splits()

    # Generate text report
    report_text = validator.generate_report(reports)
    print(report_text)

    # Save report
    with open(output_dir / 'validation_report.txt', 'w') as f:
        f.write(report_text)

    return reports


def analyze_statistics(data_root: str, output_dir: Path, num_samples: int = 100):
    """Analyze pixel value statistics for each channel"""
    print("\n" + "=" * 60)
    print("Computing Channel Statistics")
    print("=" * 60)

    data_root = Path(data_root)
    stats = {
        'train': {'DAPI': [], 'panCK': [], 'CD3': []},
        'val': {'DAPI': [], 'panCK': [], 'CD3': []},
        'test': {'DAPI': [], 'panCK': [], 'CD3': []}
    }

    for split in ['train', 'val', 'test']:
        label_dir = data_root / split / 'label'
        label_files = sorted(list(label_dir.glob('*.tif')))[:num_samples]

        print(f"\nAnalyzing {split} split ({len(label_files)} samples)...")

        for label_file in tqdm(label_files):
            img = tifffile.imread(str(label_file))

            # Each channel
            for i, channel_name in enumerate(['DAPI', 'panCK', 'CD3']):
                channel = img[:, :, i].astype(np.float32)
                stats[split][channel_name].append({
                    'mean': float(np.mean(channel)),
                    'std': float(np.std(channel)),
                    'min': float(np.min(channel)),
                    'max': float(np.max(channel)),
                    'median': float(np.median(channel))
                })

    # Aggregate statistics
    summary = {}
    for split, channels in stats.items():
        summary[split] = {}
        for channel, values in channels.items():
            means = [v['mean'] for v in values]
            summary[split][channel] = {
                'mean_mean': float(np.mean(means)),
                'mean_std': float(np.std(means)),
                'global_mean': float(np.mean([v['mean'] for v in values])),
                'global_std': float(np.mean([v['std'] for v in values]))
            }

    # Save statistics
    with open(output_dir / 'channel_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Channel Statistics Summary")
    print("=" * 60)

    for split, channels in summary.items():
        print(f"\n{split.upper()} Split:")
        for channel, stats in channels.items():
            print(f"  {channel:8s}: mean = {stats['global_mean']:.2f} ± {stats['global_std']:.2f}")

    return summary


def generate_visualizations(
    data_root: str,
    output_dir: Path,
    num_samples: int = 8,
    random_seed: int = 42
):
    """Generate input/label visualization comparisons"""
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    random.seed(random_seed)
    data_root = Path(data_root)

    for split in ['train', 'val', 'test']:
        input_dir = data_root / split / 'input'
        label_dir = data_root / split / 'label'

        input_files = sorted(list(input_dir.glob('*.tif')))
        selected_files = random.sample(input_files, min(num_samples, len(input_files)))

        fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for idx, input_file in enumerate(selected_files):
            label_file = label_dir / input_file.name

            # Load images
            input_img = tifffile.imread(str(input_file))
            label_img = tifffile.imread(str(label_file))

            # Handle grayscale input
            if len(input_img.shape) == 2:
                input_img = np.stack([input_img] * 3, axis=-1)

            # Normalize for display
            input_display = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)

            # Plot input
            axes[idx, 0].imshow(input_display)
            axes[idx, 0].set_title(f'Input: {input_file.name}')
            axes[idx, 0].axis('off')

            # Plot ground truth channels
            label_rgb = np.zeros((label_img.shape[0], label_img.shape[1], 3))
            label_rgb[:, :, 0] = label_img[:, :, 2]  # CD3 -> R
            label_rgb[:, :, 1] = label_img[:, :, 1]  # panCK -> G
            label_rgb[:, :, 2] = label_img[:, :, 0]  # DAPI -> B
            label_rgb = (label_rgb - label_rgb.min()) / (label_rgb.max() - label_rgb.min() + 1e-8)

            axes[idx, 1].imshow(label_rgb)
            axes[idx, 1].set_title('Ground Truth (mIHC)')
            axes[idx, 1].axis('off')

            # Plot individual channels
            channel_names = ['DAPI', 'panCK', 'CD3']
            for c_idx, channel_name in enumerate(channel_names):
                channel = label_img[:, :, c_idx]
                channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
                axes[idx, 2 + c_idx].imshow(channel_norm, cmap='gray' if c_idx != 0 else 'viridis')
                axes[idx, 2 + c_idx].set_title(f'{channel_name} channel')
                axes[idx, 2 + c_idx].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'visualization_{split}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization for {split} split")


def generate_channel_distributions(data_root: str, output_dir: Path, num_samples: int = 50):
    """Generate channel intensity distribution histograms"""
    print("\n" + "=" * 60)
    print("Generating Channel Distributions")
    print("=" * 60)

    data_root = Path(data_root)

    for split in ['train', 'val', 'test']:
        label_dir = data_root / split / 'label'
        label_files = sorted(list(label_dir.glob('*.tif')))[:num_samples]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        channel_names = ['DAPI', 'panCK', 'CD3']
        colors = ['blue', 'green', 'red']

        for c_idx, (channel_name, color) in enumerate(zip(channel_names, colors)):
            all_values = []
            for label_file in tqdm(label_files, desc=f'Loading {channel_name}'):
                img = tifffile.imread(str(label_file))
                channel = img[:, :, c_idx].flatten()
                all_values.extend(channel.tolist())

            axes[c_idx].hist(all_values, bins=50, color=color, alpha=0.7, density=True)
            axes[c_idx].set_title(f'{channel_name} Intensity Distribution')
            axes[c_idx].set_xlabel('Pixel Value')
            axes[c_idx].set_ylabel('Density')
            axes[c_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'distribution_{split}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved distribution for {split} split")


def document_splits(data_root: str, output_dir: Path):
    """Document the train/val/test splits"""
    print("\n" + "=" * 60)
    print("Documenting Splits")
    print("=" * 60)

    data_root = Path(data_root)

    split_info = {
        'description': 'HEMIT dataset splits as used in the original HEMIT paper',
        'random_seed': 42,
        'splits': {}
    }

    for split in ['train', 'val', 'test']:
        input_dir = data_root / split / 'input'
        files = sorted([f.name for f in input_dir.glob('*.tif')])

        split_info['splits'][split] = {
            'num_samples': len(files),
            'sample_ids': files[:10],  # First 10 as examples
            'total_files': len(files)
        }

        print(f"{split}: {len(files)} samples")

    # Save split documentation
    with open(output_dir / 'split_documentation.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\nSplit documentation saved to {output_dir / 'split_documentation.json'}")


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")

    # Check if data exists
    if not data_root.exists():
        print(f"\nError: Data root {data_root} does not exist!")
        print("Please download the HEMIT dataset from:")
        print("  https://data.mendeley.com/datasets/3gx53zm49d/1")
        return

    # Validate dataset
    validate_dataset(str(data_root), output_dir)

    # Analyze statistics
    analyze_statistics(str(data_root), output_dir)

    # Generate visualizations
    generate_visualizations(
        str(data_root),
        output_dir,
        num_samples=args.num_visualizations,
        random_seed=args.random_seed
    )

    # Generate channel distributions
    generate_channel_distributions(str(data_root), output_dir)

    # Document splits
    document_splits(str(data_root), output_dir)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
