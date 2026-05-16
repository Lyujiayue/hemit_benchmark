# HEMIT Benchmark Framework

A reproducible benchmark for **H&E to multiplex-immunohistochemistry (mIHC) image translation** on the HEMIT dataset.

## Overview

This framework implements multiple baseline and state-of-the-art methods for virtual staining of histopathology images:

| Method | Type | Reference |
|--------|------|-----------|
| pix2pix (U-Net) | Classic (Self-trained weights) | Isola et al., CVPR 2017 |
| Dual-Branch Pix2pix | Classic (Pre-trained weights reproduction) | Bian et al., arXiv:2403.18501 |
| DTR(official) | Advanced (Pre-trained weights reproduction) | Ma et al., arXiv:2509.14119 |
| DGR/DTR | Advanced (Self-trained weights) | Ma et al., arXiv:2509.14119 |
| MIPHEI-ViT | Advanced (Pre-trained weights reproduction)| Balezo et al., Comput. Biol. Med., 2026 |

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
- **Input**: H&E stained images (3 channel, RGB)
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
conda env create -f environment.yaml
conda activate hemit_benchmark
```

---

## Quick Start

### 1. Download Dataset

Download from the Mendeley link above and organize as:

```text
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
    --data_root /path/to/HEMIT \
    --output_dir ./data_prep \
    --num_visualizations 8
```

This script checks:
- **File counts per split** (Checks if sample counts match expectations)
- **Image sizes** (Validates 1024×1024 resolution for both inputs and labels)
- **Channel counts** (Ensures input is 1 or 3 channels, and label is exactly 3 channels: DAPI, panCK, CD3)
- **Pixel value ranges** (Computes min/max/mean/std global statistics)
- **Input/label filename matching** (Ensures every input image has a corresponding label)

And generates:
- `validation_report.txt` (Human-readable data integrity report)
- `channel_statistics.json` (Detailed intensity statistics for each mIHC channel)
- `split_documentation.json` (List of file names belonging to train/val/test splits)
- `visualization_{split}.png` (Input/Label multi-channel comparison images for visual inspection)
- `distribution_{split}.png` (Pixel intensity distribution histograms for each channel)

### 3. Train a Model

```bash
# pix2pix U-Net
python scripts/train_pix2pix.py \
    --config configs/pix2pix_unet.yaml \
    --data_root /path/to/HEMIT \
    --exp_name pix2pix_unet_baseline 

# DGR/DTR
python scripts/train_dgr_dtr.py \
    --config configs/dgr_dtr.yaml \
    --data_root /path/to/HEMIT \
    --exp_name dgr_dtr_baseline
```

### 4. Quantitative Evaluation

> 💡 **Note**: For `pix2pix` and `DGR/DTR`, metric results are computed and printed automatically during the training phase (see the corresponding `train` method).

For other pretrained baselines, run the following standalone script from the root directory:

```bash
# Dual-Branch
python utils/run_metrics_dual.py

# DTR (official)
python utils/run_metrics_dtr.py

# MIPHEI-ViT
python utils/run_metrics_vit.py
```

---

### 5. Figures Evaluation (Inference Figures / Failure Cases)

```bash
# pix2pix
python scripts/generate_figures_baselines.py
python scripts/mine_failures.py \
    --config configs/pix2pix_unet.yaml \
    --ckpt ./experiments/pix2pix_unet_baseline/checkpoints/best.pth \
    --data_root /path/to/HEMIT \
    --output_dir ./report_figures \
    --num_failures 4
```

```bash
# DGR/DTR
python scripts/generate_figures_advanced_dgr_dtr.py
python scripts/mine_failures.py \
    --config configs/dgr_dtr.yaml \
    --ckpt ./experiments/dgr_dtr_baseline/checkpoints/best.pth \
    --data_root /path/to/HEMIT \
    --output_dir ./report_figures \
    --num_failures 4
```

```bash
# Dual-Branch
python scripts/generate_figures_dual.py
python scripts/find_failure_cases_dual.py
```

```bash
# DTR(official)
python scripts/generate_figures_dtr.py
python scripts/find_failure_cases_dtr.py
```

```bash
# MIPHEI-ViT
python scripts/generate_figures_vit.py
python scripts/find_failure_cases_vit.py
```

---

## Project Structure

```text
/root/hemit_benchmark
├── configs
├── data
│   └── __pycache__
├── data_exploration
├── experiments
│   ├── dgr_dtr_baseline
│   │   ├── checkpoints
│   │   ├── logs
│   │   └── visualizations
│   └── pix2pix_unet_baseline
│       ├── checkpoints
│       ├── logs
│       └── visualizations
├── models
│   ├── __pycache__
│   ├── advanced
│   │   ├── MIPHEI-ViT             # git clone MIPHEI-ViT repo here
│   │   ├── __pycache__
│   │   └── dtr                    # git clone DTR repo here
│   └── baselines
│       ├── __pycache__
│       └── dual_branch_pix2pix    # git clone Dual-Branch repo here
├── report_figures
├── results
├── scripts
└── utils
    └── __pycache__
```

---

## Pretrained Weights

| Method | Download | Notes |
|--------|----------|-------|
| Dual-Branch (HEMIT) | [Google Drive](https://drive.google.com/file/d/1HNc-dj2ATN7gdAyOCy-lWe8_YQse2CTd/view) | From original paper repo |
| DTR (HEMIT) | [GitHub Release](https://github.com/birkhoffkiki/DTR/releases/download/weights/hemit_weight.pth) | Pretrained by DTR authors |

Download and place weights into the corresponding experiment `checkpoints/` directory.

---

## References

This repository is built upon the following pioneering research works and open-source projects. We sincerely thank the authors for their wonderful contributions to the community:

1. **HEMIT-DATASET**
   * Repository: [BianChang/HEMIT-DATASET](https://github.com/BianChang/HEMIT-DATASET)
   * Description: Official dataset repository for H&E to mIHC image translation benchmark.

2. **Pix2pix_DualBranch**
   * Repository: [BianChang/Pix2pix_DualBranch](https://github.com/BianChang/Pix2pix_DualBranch)
   * Description: Baseline framework for the dual-branch image-to-image translation architecture.

3. **DTR**
   * Repository: [birkhoffkiki/DTR](https://github.com/birkhoffkiki/DTR)
   * Description: Core implementation related to token/feature resolution or advanced generative baselines.

4. **MIPHEI-ViT**
   * Repository: [Sanofi-Public/MIPHEI-ViT](https://github.com/Sanofi-Public/MIPHEI-ViT)
   * Citation: *Balezo et al., MIPHEI-ViT: Multiplex immunofluorescence prediction from H&E images using ViT foundation models, Computers in Biology and Medicine, 2026.*

