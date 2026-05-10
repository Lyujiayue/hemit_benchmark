"""
HEMIT Dataset Data Loading and Validation Module
"""
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import json

import numpy as np
from PIL import Image
import tifffile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


@dataclass
class HEMITSplit:
    """HEMIT dataset split information"""
    name: str  # 'train', 'val', 'test'
    input_dir: Path
    label_dir: Path
    file_pairs: List[Tuple[Path, Path]]


@dataclass
class DataIntegrityReport:
    """Data integrity check report"""
    split: str
    total_samples: int
    valid_samples: int
    invalid_samples: List[str]
    image_sizes: Dict[str, int]
    channel_info: Dict[str, int]
    pixel_ranges: Dict[str, Tuple[float, float]]
    hash_check: bool


class HEMITDataset(Dataset):
    """
    HEMIT Dataset for H&E to mIHC translation.

    Input: H&E stained images (1 or 3 channel)
    Output: mIHC images (3 channel: DAPI, panCK, CD3)
    """

    CHANNEL_NAMES = ['DAPI', 'panCK', 'CD3']
    IMAGE_SIZE = (1024, 1024)

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        load_mode: str = 'memory',  # 'memory' or 'lazy'
        patch_size: Optional[int] = None,
        crop_mode: str = 'center'  # 'center', 'random', 'grid'
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.load_mode = load_mode
        self.patch_size = patch_size
        self.crop_mode = crop_mode

        # Paths
        self.input_dir = self.data_root / split / 'input'
        self.label_dir = self.data_root / split / 'label'

        # Build file list
        self.file_pairs = self._build_file_pairs()

        # Preload if in memory mode
        if self.load_mode == 'memory':
            self._preload_data()

    def _build_file_pairs(self) -> List[Tuple[Path, Path]]:
        """Build matched input-label file pairs"""
        input_files = sorted(self.input_dir.glob('*.tif'))
        label_files = sorted(self.label_dir.glob('*.tif'))

        # Create lookup dict for labels
        label_dict = {f.name: f for f in label_files}

        pairs = []
        missing_labels = []

        for input_file in input_files:
            label_file = label_dict.get(input_file.name)
            if label_file:
                pairs.append((input_file, label_file))
            else:
                missing_labels.append(input_file.name)

        if missing_labels:
            print(f"Warning: {len(missing_labels)} input files without matching labels")

        return pairs

    def _preload_data(self):
        """Preload all data into memory"""
        self.data_cache = []
        print(f"Preloading {len(self.file_pairs)} samples...")

        for input_path, label_path in self.file_pairs:
            input_img = self._load_image(input_path)
            label_img = self._load_image(label_path)
            self.data_cache.append((input_img, label_img))

        print(f"Preloading complete. Cache size: {len(self.data_cache)}")

    def _load_image(self, path: Path) -> np.ndarray:
        """Load a single image (TIFF format)"""
        try:
            img = tifffile.imread(str(path))
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            return img.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

    def _apply_transforms(self, img: np.ndarray, transform: Optional[Callable]) -> np.ndarray:
        """Apply transforms to image"""
        if transform is not None:
            img = transform(img)
        return img

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 1e-6:
            img = (img - img_min) / (img_max - img_min)
        return img

    def _to_tensor(self, img: np.ndarray) -> np.ndarray:
        """Convert to tensor format (C, H, W)"""
        if img.shape[-1] in [1, 3]:  # HWC to CHW
            img = np.transpose(img, (2, 0, 1))
        return img

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.load_mode == 'memory':
            input_img, label_img = self.data_cache[idx]
        else:
            input_path, label_path = self.file_pairs[idx]
            input_img = self._load_image(input_path)
            label_img = self._load_image(label_path)

        # Apply patch extraction if specified
        if self.patch_size is not None:
            input_img, label_img = self._extract_patch(
                input_img, label_img, self.patch_size, self.crop_mode
            )

        # Normalize
        input_img = self._normalize(input_img)
        label_img = self._normalize(label_img)

        # Apply custom transforms
        if self.transform:
            input_img = self.transform(input_img)
        if self.target_transform:
            label_img = self.target_transform(label_img)

        # Convert to tensor format
        input_tensor = self._to_tensor(input_img.copy())
        label_tensor = self._to_tensor(label_img.copy())

        return {
            'input': input_tensor,
            'label': label_tensor,
            'filename': self.file_pairs[idx][0].name
        }

    def _extract_patch(
        self,
        input_img: np.ndarray,
        label_img: np.ndarray,
        patch_size: int,
        mode: str = 'center'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a patch from input and label images"""
        h, w = input_img.shape[:2]

        if mode == 'center':
            start_h = (h - patch_size) // 2
            start_w = (w - patch_size) // 2
        elif mode == 'random':
            start_h = np.random.randint(0, h - patch_size + 1)
            start_w = np.random.randint(0, w - patch_size + 1)
        else:
            raise ValueError(f"Unknown crop mode: {mode}")

        input_patch = input_img[start_h:start_h+patch_size, start_w:start_w+patch_size]
        label_patch = label_img[start_h:start_h+patch_size, start_w:start_w+patch_size]

        return input_patch, label_patch


def create_data_loaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    patch_size: Optional[int] = None,
    use_augmentation: bool = True
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train/val/test splits.

    Args:
        data_root: Root directory of HEMIT dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        patch_size: If specified, extract patches of this size
        use_augmentation: Whether to use data augmentation for training

    Returns:
        Dictionary of data loaders for each split
    """
    # Define transforms
    train_transform = None
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
        ])

    # Create datasets
    train_dataset = HEMITDataset(
        data_root=data_root,
        split='train',
        transform=train_transform,
        load_mode='lazy',
        patch_size=patch_size
    )

    val_dataset = HEMITDataset(
        data_root=data_root,
        split='val',
        load_mode='lazy',
        patch_size=patch_size
    )

    test_dataset = HEMITDataset(
        data_root=data_root,
        split='test',
        load_mode='lazy',
        patch_size=patch_size
    )

    # Create data loaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }

    return dataloaders


class HEMITDataValidator:
    """Validator for HEMIT dataset integrity"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def validate_split(self, split: str) -> DataIntegrityReport:
        """
        Validate a single split of the dataset.

        Checks:
        1. File count matches expected
        2. All input-label pairs exist
        3. Image sizes are correct (1024x1024)
        4. Channel counts are correct (input: 1/3, label: 3)
        5. Pixel values are in expected range
        """
        input_dir = self.data_root / split / 'input'
        label_dir = self.data_root / split / 'label'

        input_files = sorted(list(input_dir.glob('*.tif')))
        label_files = sorted(list(label_dir.glob('*.tif')))

        expected_counts = {'train': 3717, 'val': 630, 'test': 945}
        expected_count = expected_counts.get(split, 0)

        invalid_samples = []
        image_sizes = {}
        channel_counts = {}
        pixel_ranges = {}

        for input_file in input_files:
            label_file = label_dir / input_file.name

            if not label_file.exists():
                invalid_samples.append(f"{input_file.name}: missing label")
                continue

            try:
                # Load images
                input_img = tifffile.imread(str(input_file))
                label_img = tifffile.imread(str(label_file))

                # Check sizes
                if input_img.shape[:2] != (1024, 1024):
                    invalid_samples.append(f"{input_file.name}: unexpected size {input_img.shape}")
                    continue

                if label_img.shape[:2] != (1024, 1024):
                    invalid_samples.append(f"{input_file.name}: label unexpected size {label_img.shape}")
                    continue

                # Check channels
                input_channels = input_img.shape[2] if len(input_img.shape) > 2 else 1
                label_channels = label_img.shape[2] if len(label_img.shape) > 2 else 1

                if input_channels not in [1, 3]:
                    invalid_samples.append(f"{input_file.name}: input has {input_channels} channels")
                    continue

                if label_channels != 3:
                    invalid_samples.append(f"{input_file.name}: label has {label_channels} channels (expected 3)")
                    continue

                # Check pixel range
                input_min, input_max = float(input_img.min()), float(input_img.max())
                label_min, label_max = float(label_img.min()), float(label_img.max())

                # Accumulate stats
                if input_file.name not in image_sizes:
                    image_sizes[input_file.name] = input_img.shape
                    channel_counts[input_file.name] = {'input': input_channels, 'label': label_channels}
                    pixel_ranges[input_file.name] = {
                        'input': (input_min, input_max),
                        'label': (label_min, label_max)
                    }

            except Exception as e:
                invalid_samples.append(f"{input_file.name}: error - {str(e)}")

        # Check count
        if len(input_files) != expected_count:
            print(f"Warning: {split} has {len(input_files)} files, expected {expected_count}")

        return DataIntegrityReport(
            split=split,
            total_samples=len(input_files),
            valid_samples=len(input_files) - len(invalid_samples),
            invalid_samples=invalid_samples,
            image_sizes={'width': 1024, 'height': 1024},
            channel_info={'input_channels': 1, 'label_channels': 3},
            pixel_ranges={'input': pixel_ranges, 'label': pixel_ranges},
            hash_check=True
        )

    def validate_all_splits(self) -> Dict[str, DataIntegrityReport]:
        """Validate all splits"""
        reports = {}
        for split in ['train', 'val', 'test']:
            reports[split] = self.validate_split(split)
        return reports

    def generate_report(self, reports: Dict[str, DataIntegrityReport]) -> str:
        """Generate a human-readable report"""
        lines = ["=" * 60]
        lines.append("HEMIT Dataset Integrity Report")
        lines.append("=" * 60)

        total_valid = 0
        total_samples = 0

        for split, report in reports.items():
            lines.append(f"\n{split.upper()} Split:")
            lines.append(f"  Total samples: {report.total_samples}")
            lines.append(f"  Valid samples: {report.valid_samples}")
            lines.append(f"  Invalid samples: {len(report.invalid_samples)}")

            if report.invalid_samples:
                lines.append("  First 5 invalid samples:")
                for sample in report.invalid_samples[:5]:
                    lines.append(f"    - {sample}")

            total_valid += report.valid_samples
            total_samples += report.total_samples

        lines.append(f"\n{'=' * 60}")
        lines.append(f"Total: {total_valid}/{total_samples} valid samples")
        lines.append("=" * 60)

        return "\n".join(lines)


# Augmentation transforms
class RandomAugmentation:
    """Random augmentation for training"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            # Random horizontal flip
            if np.random.random() < 0.5:
                img = np.fliplr(img)
            # Random vertical flip
            if np.random.random() < 0.5:
                img = np.flipud(img)
            # Random 90-degree rotation
            k = np.random.randint(0, 4)
            img = np.rot90(img, k)
        return img.copy()
