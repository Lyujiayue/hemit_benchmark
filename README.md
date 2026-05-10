# HEMIT Benchmark Framework

A reproducible benchmark for **H&E to multiplex-immunohistochemistry (mIHC) image translation** on the HEMIT dataset.

## Overview

This framework implements multiple baseline and state-of-the-art methods for virtual staining of histopathology images:

| Method | Type | Reference |
|--------|------|-----------|
| pix2pix (U-Net) | Classic cGAN | Isola et al., CVPR 2017 |
| pix2pix (ResNet) | Classic cGAN | Isola et al., CVPR 2017 |
| Dual-Branch Pix2pix | Paper method | Bian et al., arXiv:2403.18501 |
| DGR/DTR | Advanced (misalignment-robust) | Ma et al., arXiv:2509.14119 |

**Task**: Translate H&E stained images → 3-channel mIHC images (DAPI, panCK, CD3)

**Metrics**: SSIM, Pearson Correlation, PSNR (average + per-channel)

---

## Dataset

Download from: [Mendeley HEMIT Dataset](https://data.mendeley.com/datasets/3gx53zm49d/1)

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 3,717 | Training |
| Val | 630 | Hyperparameter tuning |
| Test | 945 | Final evaluation (NO tuning) |

- **Image size**: 1024×1024 pixels
- **Format**: TIFF
- **Input**: Grayscale H&E (1 channel)
- **Output**: mIHC (3 channels: DAPI, panCK, CD3)
- **Random seed**: 42

---

## Installation

### Option 1: pip

```bash
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate hemit_benchmark
```

### Option 3: Docker

```bash
docker build -t hemit_benchmark .
docker run --gpus all -it hemit_benchmark
```

---

## Quick Start

### 1. Download Dataset

Download from the Mendeley link above and organize as:

```
HEMIT/
├── train/
│   ├── input/   (*.tif)
│   └── label/   (*.tif)
├── val/
│   ├── input/
│   └── label/
└── test/
    ├── input/
    └── label/
```

### 2. Validate Data

```bash
python scripts/prepare_data.py \
    --data_path /path/to/HEMIT \
    --output_path ./data_prep \
    --num_vis_samples 8
```

This checks:
- File counts per split
- Image sizes (1024×1024)
- Channel counts (input: 1, label: 3)
- Pixel value ranges
- Input/label filename matching

And generates:
- `data_report.json`
- `visualizations/` (input/label comparison images)

### 3. Train a Model

```bash
# pix2pix U-Net
python scripts/train.py \
    --config configs/pix2pix_unet.yaml \
    --data_root /path/to/HEMIT \
    --exp_name pix2pix_unet

# Dual-Branch (HEMIT paper)
python scripts/train.py \
    --config configs/dual_branch.yaml \
    --data_root /path/to/HEMIT \
    --exp_name dual_branch

# DGR
python scripts/train.py \
    --config configs/dgr.yaml \
    --data_root /path/to/HEMIT \
    --exp_name dgr
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --config configs/pix2pix_unet.yaml \
    --checkpoint experiments/pix2pix_unet/checkpoints/best.pth \
    --data_root /path/to/HEMIT \
    --split test \
    --output_dir ./results
```

### 5. Run Full Benchmark

```bash
python run_benchmark.py \
    --data_root /path/to/HEMIT \
    --method all
```

---

## Project Structure

```
hemit_benchmark/
├── configs/               # YAML configs for each method
│   ├── pix2pix_unet.yaml
│   ├── pix2pix_resnet.yaml
│   ├── dual_branch.yaml
│   └── dgr.yaml
├── data/                  # Data loading
│   ├── __init__.py
│   └── dataset.py         # HEMITDataset, HEMITDataValidator
├── models/                # Model implementations
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── pix2pix.py     # UNet, ResNet, SwinT-ResNet generators + PatchGAN
│   │   └── dual_branch.py # Dual-Branch generator + discriminator
│   └── advanced/
│       ├── __init__.py
│       └── dgr.py         # DGR generator + discriminator + inference
├── utils/
│   ├── __init__.py
│   └── metrics.py         # SSIM, Pearson R, PSNR + aggregation
├── scripts/
│   ├── prepare_data.py    # Data validation & visualization
│   ├── train.py           # Unified training script
│   ├── evaluate.py        # Inference & evaluation
│   └── train_pix2pix.py   # Standalone pix2pix trainer
├── experiments/           # Training logs, checkpoints, visualizations
├── results/               # Evaluation results
├── run_benchmark.py       # One-command benchmark runner
├── requirements.txt
├── environment.yml
├── Dockerfile
└── README.md
```

---

## Pretrained Weights

| Method | Download | Notes |
|--------|----------|-------|
| Dual-Branch (HEMIT) | [Google Drive](https://drive.google.com/file/d/1HNc-dj2ATN7gdAyOCy-lWe8_YQse2CTd/view) | From original paper repo |
| DGR (HEMIT) | [GitHub Release](https://github.com/birkhoffkiki/DTR/releases/download/weights/hemit_weight.pth) | Pretrained by DTR authors |

Download and place in `checkpoints/` directory.

---

## Training Configuration Summary

| Parameter | pix2pix | Dual-Branch | DGR |
|-----------|---------|-------------|-----|
| Batch size | 2 | 2 | 2 |
| Learning rate | 2e-4 | 3e-5 | 2e-4 |
| λ_L1 | 100 | 30 | 100 |
| Epochs | 100 | 80 | 100 |
| LR schedule | step@50 | step@30 | step@50 |
| Generator | UNet/ResNet | Dual-Branch | DGR |

---

## Results Table Format

Final results are reported as:

```
| Metric | DAPI | panCK | CD3 | Average |
|--------|------|-------|-----|---------|
| SSIM   | ...  | ...   | ... | ...     |
| Pearson| ...  | ...   | ... | ...     |
| PSNR   | ...  | ...   | ... | ...     |
```

Both **mean ± std** are reported across all test images.

---

## Key Design Decisions

### Multi-channel Output
All methods output 3-channel mIHC images directly. The discriminator receives the concatenation of the generated image and the input H&E image as conditioning (4-channel input for PatchGAN).

### Data Normalization
- H&E input: normalized to [0, 1] per-image (min-max)
- mIHC label: normalized to [0, 1] per-image (min-max)
- Model operates in [-1, 1] range via Tanh output

### Evaluation
Metrics are computed per-channel (DAPI, panCK, CD3) and averaged. The `TINY=1e-15` constant is added to avoid numerical issues in Pearson correlation computation.

---

## Hardware Requirements

| Configuration | GPU Memory | Notes |
|---------------|-----------|-------|
| Min (1024×1024, batch=1) | ~8 GB | Slow training |
| Recommended (batch=2) | ~16 GB | Good balance |
| Optimal (batch=4, gradient checkpointing) | ~24 GB | Fast training |

---

## Troubleshooting

### Out of Memory on 1024×1024 images
- Reduce batch size to 1
- Enable gradient checkpointing (coming soon)
- Use patch-based training with smaller patches

### Dataset not found
- Ensure directory structure matches: `HEMIT/{train,val,test}/{input,label}/`
- Check that `.tif` files exist in both `input/` and `label/` directories

### Pretrained weight download fails
- Use `wget` or browser for Google Drive links
- For DGR: direct GitHub URL should work reliably

---

## References

1. Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017.
2. Bian et al., "HEMIT: H&E to Multiplex-immunohistochemistry Image Translation with Dual-Branch Pix2pix Generator", arXiv:2403.18501, 2024.
3. Ma et al., "Generative AI for Misalignment-Resistant Virtual Staining to Accelerate Histopathology Workflows", arXiv:2509.14119, 2024.
