#!/bin/bash
echo "Setting up environment for CASSPR with RTX 3090 Ti..."
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 设置GPU架构（支持RTX 3090 Ti的8.6）
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

# 激活conda环境
conda activate casspr

# 设置OpenMP线程数
export OMP_NUM_THREADS=12

echo "CUDA version:"
nvcc --version
echo "Environment ready for CASSPR"
