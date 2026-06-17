#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing CUDA Toolkit for RTX 5050 ==="

# Check if nvidia-smi works (WSL2 GPU passthrough)
if nvidia-smi &>/dev/null; then
    echo "NVIDIA driver detected via WSL2 passthrough:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "ERROR: nvidia-smi not found. Ensure NVIDIA drivers are installed on Windows."
    echo "Download from: https://www.nvidia.com/drivers"
    exit 1
fi

# Install CUDA keyring
if ! dpkg -l cuda-keyring &>/dev/null; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
fi

# Install CUDA toolkit
sudo apt-get install -y -qq cuda-toolkit-12-6

# Environment setup
CUDA_EXPORTS='
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
'

if ! grep -q "cuda/bin" ~/.bashrc; then
    echo "$CUDA_EXPORTS" >> ~/.bashrc
fi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo ""
echo "=== Verification ==="
echo "nvidia-smi:"
nvidia-smi
echo ""
echo "nvcc version:"
nvcc --version || echo "nvcc not yet in PATH — restart shell"
echo ""
echo "=== KAN-10 COMPLETE ==="
