FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /workspace

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch first (CUDA 12.1)
RUN pip install --no-cache-dir \
    torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.3 \
    scipy==1.12.0 \
    pandas==2.1.4 \
    scikit-image==0.22.0 \
    imageio==2.33.1 \
    pillow==10.2.0 \
    opencv-python==4.9.0.80 \
    matplotlib==3.8.2 \
    seaborn==0.13.1 \
    tensorboard==2.15.1 \
    einops==0.7.0 \
    tifffile==2024.1.30 \
    tqdm==4.66.1 \
    pyyaml==6.0.1 \
    ipython

# Copy project
COPY . /workspace/hemit_benchmark/

WORKDIR /workspace/hemit_benchmark

# Download pretrained weights
RUN python -c "\
import os; \
os.makedirs('checkpoints', exist_ok=True); \
# DGR HEMIT pretrained weights
import urllib.request; \
urllib.request.urlretrieve(\
    'https://github.com/birkhoffkiki/DTR/releases/download/weights/hemit_weight.pth',\
    'checkpoints/dgr_hemit_pretrained.pth'\
); \
print('Pretrained weights downloaded.')"

ENV PYTHONPATH=/workspace/hemit_benchmark:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "scripts/train.py", "--help"]
