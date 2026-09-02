#!/bin/bash
# ==============================================================================
# THSA-2B V1: GOOGLE COLAB ENVIRONMENT SETUP SCRIPT
# ==============================================================================
set -e

echo "================================================================================"
echo "THSA-2B V1: Initializing Google Colab GPU Environment"
echo "================================================================================"

# 1. Check Python & GPU
python3 --version
nvidia-smi

# 2. Upgrade pip and install exact required PyTorch/Transformers ecosystem packages
pip install --upgrade pip
pip install \
    torch>=2.1.0 \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    datasets>=2.18.0 \
    sentencepiece>=0.2.0 \
    psutil>=5.9.0 \
    accelerate>=0.29.0 \
    safetensors>=0.4.2

# 3. Verify PyTorch CUDA Availability
python3 -c "
import torch
print(f'PyTorch Version:  {torch.__version__}')
print(f'CUDA Available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device Name:  {torch.cuda.get_device_name(0)}')
    print(f'GPU Total VRAM:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
    print(f'BF16 Supported:   {torch.cuda.is_bf16_supported()}')
else:
    print('[WARNING] CUDA is NOT available! Please switch Colab runtime to GPU.')
"

echo "================================================================================"
echo "[SUCCESS] Google Colab Environment Setup Complete."
echo "================================================================================"
