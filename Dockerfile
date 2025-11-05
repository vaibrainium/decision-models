FROM ghcr.io/walkerlab/docker-pytorch-jupyter-cuda:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-22.04

LABEL maintainer="vaibrainium (vaibhavt459@gmail.com)"

COPY . /src/

# --- install CUDA 12.1 runtime ---
RUN apt-get update -y && \
    apt-get install -y wget gnupg2 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-runtime-12-1 && \
    rm -rf /var/lib/apt/lists/*

# --- upgrade PyTorch to CUDA 12.1 build ---
RUN pip install --upgrade pip && \
    pip uninstall -y torch torchvision torchaudio && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r /src/requirements.txt && \
    pip install -e /src/.

# --- install fish shell (non-interactive, cleanly) ---
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    apt-add-repository ppa:fish-shell/release-3 -y && \
    apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y fish && \
    chsh -s $(which fish) && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
