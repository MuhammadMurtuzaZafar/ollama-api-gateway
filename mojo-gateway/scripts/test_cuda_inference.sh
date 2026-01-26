#!/bin/bash
# EdgeLLM CUDA Text Generation Test - Lightning.ai T4 GPU
set -e

echo "=============================================="
echo "EdgeLLM CUDA Text Generation Test"
echo "=============================================="
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please run on a CUDA-enabled system."
    exit 1
fi

echo "CUDA Version:"
nvcc --version | head -4
echo ""

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Navigate to project
cd /teamspace/studios/this_studio/mojo-gateway 2>/dev/null || cd ~/mojo-gateway || cd .

echo "Working directory: $(pwd)"
echo ""

# Build INT8 CUDA kernel for T4
echo "Building INT8 Flash Attention kernel for T4..."
cd src/kernels/cuda
make clean 2>/dev/null || true
make t4 int8
echo ""

# Verify library
echo "Checking built library:"
ls -la ../../../lib/libflash_attention_int8.so
echo ""

# Go back to root
cd ../../..

# Download test model if needed
if [ ! -f "stories15M.bin" ]; then
    echo "Downloading stories15M model..."
    wget -q https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
fi

if [ ! -f "tokenizer.bin" ]; then
    echo "Downloading tokenizer..."
    wget -q https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
fi

echo "Model files:"
ls -la stories15M.bin tokenizer.bin
echo ""

# Build Mojo inference
echo "Building Mojo inference engine..."
pixi run mojo build -O3 src/edgellm_inference.mojo -o bin/edgellm
echo ""

# Test CPU first (baseline)
echo "=============================================="
echo "TEST 1: CPU Text Generation (baseline)"
echo "=============================================="
./bin/edgellm stories15M.bin -z tokenizer.bin -i "Once upon a time" -n 100
echo ""

# Test CUDA
echo "=============================================="
echo "TEST 2: CUDA Text Generation (stateless attention)"
echo "=============================================="
./bin/edgellm stories15M.bin -z tokenizer.bin -i "Once upon a time" -n 100 --cuda --lib ./lib
echo ""

echo "=============================================="
echo "Test Complete!"
echo "=============================================="
