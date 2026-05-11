"""
HEMIT Evaluation Metrics Module

Computes SSIM, Pearson Correlation, and PSNR for mIHC image translation evaluation.
"""
import torch
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json

import numpy as np
import tifffile
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter
import csv


@dataclass
class ChannelMetrics:
    """Metrics for a single channel"""
    ssim: float
    pearson: float
    psnr: float


@dataclass
class ImageMetrics:
    """Metrics for a single image (all 3 channels)"""
    filename: str
    dapi: ChannelMetrics
    panck: ChannelMetrics
    cd3: ChannelMetrics
    average_ssim: float = field(init=False)
    average_pearson: float = field(init=False)
    average_psnr: float = field(init=False)

    def __post_init__(self):
        self.average_ssim = (self.dapi.ssim + self.panck.ssim + self.cd3.ssim) / 3
        self.average_pearson = (self.dapi.pearson + self.panck.pearson + self.cd3.pearson) / 3
        self.average_psnr = (self.dapi.psnr + self.panck.psnr + self.cd3.psnr) / 3

    def to_dict(self) -> Dict:
        return {
            'filename': self.filename,
            'dapi_ssim': self.dapi.ssim,
            'panck_ssim': self.panck.ssim,
            'cd3_ssim': self.cd3.ssim,
            'average_ssim': self.average_ssim,
            'dapi_pearson': self.dapi.pearson,
            'panck_pearson': self.panck.pearson,
            'cd3_pearson': self.cd3.pearson,
            'average_pearson': self.average_pearson,
            'dapi_psnr': self.dapi.psnr,
            'panck_psnr': self.panck.psnr,
            'cd3_psnr': self.cd3.psnr,
            'average_psnr': self.average_psnr,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all images"""
    num_samples: int
    # Channel-wise averages
    dapi_ssim_mean: float
    dapi_ssim_std: float
    panck_ssim_mean: float
    panck_ssim_std: float
    cd3_ssim_mean: float
    cd3_ssim_std: float
    average_ssim_mean: float
    average_ssim_std: float

    dapi_pearson_mean: float
    dapi_pearson_std: float
    panck_pearson_mean: float
    panck_pearson_std: float
    cd3_pearson_mean: float
    cd3_pearson_std: float
    average_pearson_mean: float
    average_pearson_std: float

    dapi_psnr_mean: float
    dapi_psnr_std: float
    panck_psnr_mean: float
    panck_psnr_std: float
    cd3_psnr_mean: float
    cd3_psnr_std: float
    average_psnr_mean: float
    average_psnr_std: float

    def to_dict(self) -> Dict:
        return {
            'num_samples': self.num_samples,
            'ssim': {
                'DAPI': {'mean': self.dapi_ssim_mean, 'std': self.dapi_ssim_std},
                'panCK': {'mean': self.panck_ssim_mean, 'std': self.panck_ssim_std},
                'CD3': {'mean': self.cd3_ssim_mean, 'std': self.cd3_ssim_std},
                'Average': {'mean': self.average_ssim_mean, 'std': self.average_ssim_std}
            },
            'pearson': {
                'DAPI': {'mean': self.dapi_pearson_mean, 'std': self.dapi_pearson_std},
                'panCK': {'mean': self.panck_pearson_mean, 'std': self.panck_pearson_std},
                'CD3': {'mean': self.cd3_pearson_mean, 'std': self.cd3_pearson_std},
                'Average': {'mean': self.average_pearson_mean, 'std': self.average_pearson_std}
            },
            'psnr': {
                'DAPI': {'mean': self.dapi_psnr_mean, 'std': self.dapi_psnr_std},
                'panCK': {'mean': self.panck_psnr_mean, 'std': self.panck_psnr_std},
                'CD3': {'mean': self.cd3_psnr_mean, 'std': self.cd3_psnr_std},
                'Average': {'mean': self.average_psnr_mean, 'std': self.average_psnr_std}
            }
        }


class MetricsCalculator:
    """Calculator for HEMIT evaluation metrics"""

    CHANNEL_NAMES = ['DAPI', 'panCK', 'CD3']
    TINY = 1e-15  # Small constant to avoid numerical issues

    def __init__(self, data_range: Optional[float] = None):
        self.data_range = data_range

    def compute_channel_metrics(
        self,
        real_channel: np.ndarray,
        fake_channel: np.ndarray,
        channel_name: str
    ) -> ChannelMetrics:
        """
        Compute SSIM, Pearson correlation, and PSNR for a single channel.

        Args:
            real_channel: Ground truth channel (H, W)
            fake_channel: Predicted channel (H, W)
            channel_name: Name of the channel

        Returns:
            ChannelMetrics object
        """
        # Ensure float type
        real_channel = real_channel.astype(np.float64)
        fake_channel = fake_channel.astype(np.float64)

        # Add tiny value to avoid zero values affecting correlation
        real_flat = real_channel.flatten().copy()
        fake_flat = fake_channel.flatten().copy()
        real_flat[0] += self.TINY
        fake_flat[0] += self.TINY

        # Compute SSIM
        if self.data_range is not None:
            data_range = self.data_range
        else:
            data_range = real_channel.max() - real_channel.min()

        ssim_value = ssim(
            real_channel,
            fake_channel,
            data_range=data_range,
            channel_axis=None
        )

        # Compute Pearson correlation
        pearson_value, _ = pearsonr(real_flat, fake_flat)

        # Compute PSNR
        psnr_value = psnr(
            real_channel,
            fake_channel,
            data_range=data_range
        )

        return ChannelMetrics(
            ssim=ssim_value,
            pearson=pearson_value,
            psnr=psnr_value
        )

    def compute_image_metrics(
        self,
        real_image: np.ndarray,
        fake_image: np.ndarray,
        filename: str = ''
    ) -> ImageMetrics:
        """
        Compute metrics for all channels of a single image.

        Args:
            real_image: Ground truth mIHC image (H, W, 3)
            fake_image: Predicted mIHC image (H, W, 3)
            filename: Name of the image file

        Returns:
            ImageMetrics object
        """
        metrics = []
        for i, channel_name in enumerate(self.CHANNEL_NAMES):
            real_channel = real_image[:, :, i]
            fake_channel = fake_image[:, :, i]
            channel_metrics = self.compute_channel_metrics(
                real_channel, fake_channel, channel_name
            )
            metrics.append(channel_metrics)

        return ImageMetrics(
            filename=filename,
            dapi=metrics[0],
            panck=metrics[1],
            cd3=metrics[2]
        )

    def compute_batch_metrics(
        self,
        real_images: np.ndarray,
        fake_images: np.ndarray,
        filenames: Optional[List[str]] = None
    ) -> List[ImageMetrics]:
        """
        Compute metrics for a batch of images.

        Args:
            real_images: Batch of ground truth images (B, H, W, 3) or (B, 3, H, W)
            fake_images: Batch of predicted images (B, H, W, 3) or (B, 3, H, W)
            filenames: List of filenames for each image

        Returns:
            List of ImageMetrics objects
        """
        # Convert CHW to HWC if needed
        if real_images.shape[1] == 3:
            real_images = np.transpose(real_images, (0, 2, 3, 1))
        if fake_images.shape[1] == 3:
            fake_images = np.transpose(fake_images, (0, 2, 3, 1))

        batch_size = real_images.shape[0]
        if filenames is None:
            filenames = [f'image_{i}' for i in range(batch_size)]

        results = []
        for i in range(batch_size):
            img_metrics = self.compute_image_metrics(
                real_images[i],
                fake_images[i],
                filenames[i]
            )
            results.append(img_metrics)

        return results

    def aggregate_metrics(self, image_metrics: List[ImageMetrics]) -> AggregateMetrics:
        """
        Aggregate metrics across all images.

        Args:
            image_metrics: List of ImageMetrics objects

        Returns:
            AggregateMetrics object with mean and std for each metric
        """
        num_samples = len(image_metrics)

        # Extract individual metrics
        dapi_ssim = [m.dapi.ssim for m in image_metrics]
        panck_ssim = [m.panck.ssim for m in image_metrics]
        cd3_ssim = [m.cd3.ssim for m in image_metrics]
        avg_ssim = [m.average_ssim for m in image_metrics]

        dapi_pearson = [m.dapi.pearson for m in image_metrics]
        panck_pearson = [m.panck.pearson for m in image_metrics]
        cd3_pearson = [m.cd3.pearson for m in image_metrics]
        avg_pearson = [m.average_pearson for m in image_metrics]

        dapi_psnr = [m.dapi.psnr for m in image_metrics]
        panck_psnr = [m.panck.psnr for m in image_metrics]
        cd3_psnr = [m.cd3.psnr for m in image_metrics]
        avg_psnr = [m.average_psnr for m in image_metrics]

        return AggregateMetrics(
            num_samples=num_samples,
            dapi_ssim_mean=np.mean(dapi_ssim), dapi_ssim_std=np.std(dapi_ssim),
            panck_ssim_mean=np.mean(panck_ssim), panck_ssim_std=np.std(panck_ssim),
            cd3_ssim_mean=np.mean(cd3_ssim), cd3_ssim_std=np.std(cd3_ssim),
            average_ssim_mean=np.mean(avg_ssim), average_ssim_std=np.std(avg_ssim),
            dapi_pearson_mean=np.mean(dapi_pearson), dapi_pearson_std=np.std(dapi_pearson),
            panck_pearson_mean=np.mean(panck_pearson), panck_pearson_std=np.std(panck_pearson),
            cd3_pearson_mean=np.mean(cd3_pearson), cd3_pearson_std=np.std(cd3_pearson),
            average_pearson_mean=np.mean(avg_pearson), average_pearson_std=np.std(avg_pearson),
            dapi_psnr_mean=np.mean(dapi_psnr), dapi_psnr_std=np.std(dapi_psnr),
            panck_psnr_mean=np.mean(panck_psnr), panck_psnr_std=np.std(panck_psnr),
            cd3_psnr_mean=np.mean(cd3_psnr), cd3_psnr_std=np.std(cd3_psnr),
            average_psnr_mean=np.mean(avg_psnr), average_psnr_std=np.std(avg_psnr)
        )


class HEMITEvaluator:
    """
    Main evaluator class for HEMIT benchmark.

    Can evaluate from:
    1. Image directories (for inference outputs)
    2. Direct numpy arrays (for training validation)
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.calculator = MetricsCalculator()

    def evaluate_from_directory(
        self,
        real_dir: str,
        fake_dir: str,
        output_csv: Optional[str] = None
    ) -> Tuple[List[ImageMetrics], AggregateMetrics]:
        """
        Evaluate predictions from image directories.

        Args:
            real_dir: Directory containing ground truth images (real_B_*.tif)
            fake_dir: Directory containing predicted images (fake_B_*.tif)
            output_csv: If provided, save results to CSV file

        Returns:
            Tuple of (per-image metrics, aggregate metrics)
        """
        real_dir = Path(real_dir)
        fake_dir = Path(fake_dir)

        fake_files = sorted(fake_dir.glob('fake_B_*.tif'))

        image_metrics = []
        for fake_file in fake_files:
            # Get corresponding real file name
            base_name = fake_file.stem.replace('fake_B_', '')
            real_file = real_dir / f'real_B_{base_name}.tif'

            if not real_file.exists():
                print(f"Warning: Real file {real_file} not found for {fake_file.name}")
                continue

            # Load images
            real_img = tifffile.imread(str(real_file))
            fake_img = tifffile.imread(str(fake_file))

            # Compute metrics
            metrics = self.calculator.compute_image_metrics(
                real_img, fake_img, base_name
            )
            image_metrics.append(metrics)

        # Aggregate
        aggregate = self.calculator.aggregate_metrics(image_metrics)

        # Save to CSV if requested
        if output_csv:
            self._save_to_csv(image_metrics, output_csv)

        return image_metrics, aggregate

    def _save_to_csv(self, metrics: List[ImageMetrics], output_path: str):
        """Save per-image metrics to CSV file"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'filename',
                'dapi_ssim', 'panck_ssim', 'cd3_ssim', 'average_ssim',
                'dapi_pearson', 'panck_pearson', 'cd3_pearson', 'average_pearson',
                'dapi_psnr', 'panck_psnr', 'cd3_psnr', 'average_psnr'
            ])

            for m in metrics:
                writer.writerow([
                    m.filename,
                    f'{m.dapi.ssim:.4f}', f'{m.panck.ssim:.4f}', f'{m.cd3.ssim:.4f}', f'{m.average_ssim:.4f}',
                    f'{m.dapi.pearson:.4f}', f'{m.panck.pearson:.4f}', f'{m.cd3.pearson:.4f}', f'{m.average_pearson:.4f}',
                    f'{m.dapi.psnr:.2f}', f'{m.panck.psnr:.2f}', f'{m.cd3.psnr:.2f}', f'{m.average_psnr:.2f}'
                ])


def print_metrics_table(aggregate: AggregateMetrics, method_name: str = ''):
    """Print a formatted metrics table"""
    d = aggregate.to_dict()

    header = f"\n{'='*70}"
    if method_name:
        header += f"\n{ method_name} Results"
    header += f"\n{'='*70}"

    table = f"""
{header}
| Metric | DAPI          | panCK         | CD3           | Average       |
|--------|---------------|---------------|---------------|---------------|
| SSIM   | {d['ssim']['DAPI']['mean']:.4f} ± {d['ssim']['DAPI']['std']:.4f} | {d['ssim']['panCK']['mean']:.4f} ± {d['ssim']['panCK']['std']:.4f} | {d['ssim']['CD3']['mean']:.4f} ± {d['ssim']['CD3']['std']:.4f} | {d['ssim']['Average']['mean']:.4f} ± {d['ssim']['Average']['std']:.4f} |
| Pearson| {d['pearson']['DAPI']['mean']:.4f} ± {d['pearson']['DAPI']['std']:.4f} | {d['pearson']['panCK']['mean']:.4f} ± {d['pearson']['panCK']['std']:.4f} | {d['pearson']['CD3']['mean']:.4f} ± {d['pearson']['CD3']['std']:.4f} | {d['pearson']['Average']['mean']:.4f} ± {d['pearson']['Average']['std']:.4f} |
| PSNR   | {d['psnr']['DAPI']['mean']:.2f} ± {d['psnr']['DAPI']['std']:.2f} | {d['psnr']['panCK']['mean']:.2f} ± {d['psnr']['panCK']['std']:.2f} | {d['psnr']['CD3']['mean']:.2f} ± {d['psnr']['CD3']['std']:.2f} | {d['psnr']['Average']['mean']:.2f} ± {d['psnr']['Average']['std']:.2f} |

Samples evaluated: {aggregate.num_samples}
{'='*70}
"""
    print(table)
    return table


def save_results_json(aggregate: AggregateMetrics, output_path: str):
    """Save aggregate metrics to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(aggregate.to_dict(), f, indent=2)


# Perceptual metrics (optional bonus)
class PerceptualMetrics:
    """Optional perceptual metrics: LPIPS, FID, DISTS"""

    def __init__(self):
        self.lpips_model = None
        self.dists_model = None

    def compute_lpips(self, real_img: np.ndarray, fake_img: np.ndarray) -> float:
        """
        Compute Learned Perceptual Image Patch Similarity (LPIPS).
        Requires lpips package: pip install lpips
        """
        try:
            import lpips
            if self.lpips_model is None:
                self.lpips_model = lpips.LPIPS(net='alex')

            # Preprocess images
            real_tensor = self._preprocess_for_lpips(real_img)
            fake_tensor = self._preprocess_for_lpips(fake_img)

            with torch.no_grad():
                distance = self.lpips_model(real_tensor, fake_tensor)

            return float(distance.item())
        except ImportError:
            print("LPIPS not available. Install with: pip install lpips")
            return 0.0

    def compute_dists(self, real_img: np.ndarray, fake_img: np.ndarray) -> float:
        """
        compute Deep Image Structure and Texture Similarity (DISTS).
        Requires piq package: pip install piq
        """
        try:
            from piq import DISTS
            if self.dists_model is None:
                self.dists_model = DISTS()

            real_tensor = self._preprocess_for_piq(real_img)
            fake_tensor = self._preprocess_for_piq(fake_img)

            with torch.no_grad():
                distance = self.dists_model(real_tensor, fake_tensor)

            return float(distance.item())
        except ImportError:
            print("DISTS not available. Install with: pip install piq")
            return 0.0

    def _preprocess_for_lpips(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for LPIPS"""
        import torch
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) if len(img.shape) == 2 else torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img = img.float() / 255.0 if img.max() > 1.0 else img
        return img

    def _preprocess_for_piq(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image for PIQ metrics"""
        import torch
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return img
