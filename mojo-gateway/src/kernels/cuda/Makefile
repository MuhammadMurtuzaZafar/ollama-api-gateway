# EdgeLLM CUDA Kernel Build System
# Builds shared libraries for NVIDIA GPUs

NVCC = nvcc
NVCC_FLAGS_COMMON = -O3 -Xcompiler -fPIC -Xcompiler -Wall

# CUDA architecture flags
# Jetson Nano: sm_53
# Jetson Xavier/Orin: sm_72, sm_87
# RTX 20 series: sm_75
# RTX 30 series: sm_86
# RTX 40 series: sm_89
CUDA_ARCH ?= -gencode arch=compute_53,code=sm_53 \
             -gencode arch=compute_72,code=sm_72 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_86,code=sm_86

# Platform detection
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    SHARED_EXT = dylib
    SHARED_FLAGS = -Xlinker -dynamiclib
else
    SHARED_EXT = so
    SHARED_FLAGS = -shared
endif

NVCC_FLAGS = $(NVCC_FLAGS_COMMON) $(CUDA_ARCH)

# Output directories
LIB_DIR = ../../../lib
BIN_DIR = ../../../bin

# Targets
TARGET = $(LIB_DIR)/libtmac_kernel_cuda.$(SHARED_EXT)
STATIC_TARGET = $(LIB_DIR)/libtmac_kernel_cuda.a
FA_TARGET = $(LIB_DIR)/libflash_attention.$(SHARED_EXT)
INT8_TARGET = $(LIB_DIR)/libflash_attention_int8.$(SHARED_EXT)
FA2_TARGET = $(LIB_DIR)/libflash_attention_v2.$(SHARED_EXT)

# New inference pipeline targets
RMSNORM_TARGET = $(LIB_DIR)/librmsnorm_kernel.$(SHARED_EXT)
FFN_TARGET = $(LIB_DIR)/libffn_kernel.$(SHARED_EXT)
EMBEDDINGS_TARGET = $(LIB_DIR)/libembeddings_kernel.$(SHARED_EXT)
SAMPLING_TARGET = $(LIB_DIR)/libsampling_kernel.$(SHARED_EXT)
LINEAR_TARGET = $(LIB_DIR)/liblinear_projection.$(SHARED_EXT)
RESIDUAL_TARGET = $(LIB_DIR)/libresiudal_add.$(SHARED_EXT)
PIPELINE_TARGET = $(LIB_DIR)/libinference_pipeline.$(SHARED_EXT)
INFERENCE_TARGET = $(LIB_DIR)/libedgellm_inference.$(SHARED_EXT)

# cuBLAS full GPU inference (key for 400+ tok/s)
CUBLAS_TARGET = $(LIB_DIR)/libcublas_matmul.$(SHARED_EXT)

SOURCES = tmac_kernel.cu flash_attention.cu
OBJECTS = tmac_kernel.o flash_attention.o
HEADERS = tmac_kernel_cuda.h flash_attention.h flash_attention_int8.h inference_kernels.h

# New inference kernel objects
RMSNORM_OBJ = rmsnorm_kernel.o
FFN_OBJ = ffn_kernel.o
EMBEDDINGS_OBJ = embeddings_kernel.o
SAMPLING_OBJ = sampling_kernel.o
LINEAR_OBJ = linear_projection.o
RESIDUAL_OBJ = residual_add.o
PIPELINE_OBJ = inference_pipeline.o

.PHONY: all clean test install dirs check info

# Default target
all: check dirs $(TARGET)

# Check for NVCC
check:
	@which nvcc > /dev/null 2>&1 || (echo "Error: nvcc not found. Install CUDA Toolkit." && exit 1)

dirs:
	@mkdir -p $(LIB_DIR) $(BIN_DIR)

$(TARGET): $(OBJECTS)
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built CUDA kernel: $(TARGET)"

$(STATIC_TARGET): $(OBJECTS)
	ar rcs $@ $^
	@echo "Built static library: $(STATIC_TARGET)"

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Separate Flash Attention library
$(FA_TARGET): flash_attention.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built Flash Attention: $(FA_TARGET)"

# Build all libraries including Flash Attention
flash: dirs $(TARGET) $(FA_TARGET)
	@echo "Built all CUDA kernels including Flash Attention"

# INT8 Tensor Core Flash Attention (requires sm_75+ for WMMA)
# Use separate arch flags that only include Turing+ GPUs
INT8_CUDA_ARCH = -gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_86,code=sm_86 \
                 -gencode arch=compute_89,code=sm_89
INT8_NVCC_FLAGS = $(NVCC_FLAGS_COMMON) $(INT8_CUDA_ARCH)

flash_attention_int8.o: flash_attention_int8.cu flash_attention_int8.h
	$(NVCC) $(INT8_NVCC_FLAGS) -c $< -o $@

$(INT8_TARGET): flash_attention_int8.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built INT8 Flash Attention: $(INT8_TARGET)"

# Build INT8 Tensor Core kernels
int8: dirs flash_attention_int8.o $(INT8_TARGET)
	@echo "Built INT8 Tensor Core Flash Attention"

# Build all including INT8
all-int8: dirs $(TARGET) $(FA_TARGET) $(INT8_TARGET)
	@echo "Built all CUDA kernels including INT8 Tensor Core Flash Attention"

# FlashAttention-2 (tiled with online softmax)
flash_attention_v2.o: flash_attention_v2.cu flash_attention_v2.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(FA2_TARGET): flash_attention_v2.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built FlashAttention-2: $(FA2_TARGET)"

# Build FlashAttention-2
fa2: dirs flash_attention_v2.o $(FA2_TARGET)
	@echo "Built FlashAttention-2 (tiled, online softmax)"

# FlashAttention-2 Optimized (WMMA + GPU-Resident + FlashDecoding++)
FA2_OPT_TARGET = $(LIB_DIR)/libflash_attention_v2_optimized.$(SHARED_EXT)

flash_attention_v2_optimized.o: flash_attention_v2_optimized.cu flash_attention_v2_optimized.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(FA2_OPT_TARGET): flash_attention_v2_optimized.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built FlashAttention-2 Optimized: $(FA2_OPT_TARGET)"

# Build Optimized FlashAttention-2
fa2-opt: dirs flash_attention_v2_optimized.o $(FA2_OPT_TARGET)
	@echo "Built FlashAttention-2 Optimized (WMMA, GPU-Resident, Split-K)"

# Test optimized FA2
test-fa2-opt: dirs flash_attention_v2_optimized.o
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/test_flash_attention_v2_optimized test_flash_attention_v2_optimized.cu flash_attention_v2_optimized.o -lcudart
	@echo "Running Optimized FlashAttention-2 tests..."
	$(BIN_DIR)/test_flash_attention_v2_optimized

# FlashAttention-2 Multi-GPU (tensor parallel with NCCL)
FA2_MG_TARGET = $(LIB_DIR)/libflash_attention_v2_multi_gpu.$(SHARED_EXT)

flash_attention_v2_multi_gpu.o: flash_attention_v2_multi_gpu.cu flash_attention_v2_multi_gpu.h flash_attention_v2.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(FA2_MG_TARGET): flash_attention_v2_multi_gpu.o flash_attention_v2.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lnccl -lpthread -lcudart
	@echo "Built Multi-GPU FlashAttention-2: $(FA2_MG_TARGET)"

# Build Multi-GPU FlashAttention-2
fa2-multi-gpu: dirs flash_attention_v2.o flash_attention_v2_multi_gpu.o $(FA2_MG_TARGET)
	@echo "Built Multi-GPU FlashAttention-2 (tensor parallel, NCCL)"

# Multi-GPU test
test-fa2-multi-gpu: dirs flash_attention_v2.o flash_attention_v2_multi_gpu.o
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/test_flash_attention_v2_multi_gpu test_flash_attention_v2_multi_gpu.cu flash_attention_v2.o flash_attention_v2_multi_gpu.o -lnccl -lpthread -lcudart
	@echo "Running Multi-GPU FlashAttention-2 tests..."
	$(BIN_DIR)/test_flash_attention_v2_multi_gpu

# Build all attention variants
all-attention: dirs $(FA_TARGET) $(INT8_TARGET) $(FA2_TARGET) $(FA2_OPT_TARGET)
	@echo "Built all attention kernels (FP32, INT8, FA2, FA2-Optimized)"

# Build all attention variants including multi-GPU
all-attention-multi-gpu: dirs $(FA_TARGET) $(INT8_TARGET) $(FA2_TARGET) $(FA2_MG_TARGET) $(FA2_OPT_TARGET)
	@echo "Built all attention kernels (FP32, INT8, FA2, Multi-GPU, Optimized)"

# =============================================================================
# Inference Pipeline Kernels (RMSNorm, FFN, Embeddings, Sampling)
# =============================================================================

# RMSNorm kernel
rmsnorm_kernel.o: rmsnorm_kernel.cu rmsnorm_kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(RMSNORM_TARGET): rmsnorm_kernel.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built RMSNorm kernel: $(RMSNORM_TARGET)"

rmsnorm: dirs $(RMSNORM_TARGET)
	@echo "Built RMSNorm kernel"

# FFN/MLP kernel (SwiGLU) - uses __dp4a which requires sm_61+
# Define DP4A-compatible architectures
DP4A_CUDA_ARCH = -gencode arch=compute_61,code=sm_61 \
                 -gencode arch=compute_70,code=sm_70 \
                 -gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_86,code=sm_86
DP4A_NVCC_FLAGS = $(NVCC_FLAGS_COMMON) $(DP4A_CUDA_ARCH)

ffn_kernel.o: ffn_kernel.cu ffn_kernel.h
	$(NVCC) $(DP4A_NVCC_FLAGS) -c $< -o $@

$(FFN_TARGET): ffn_kernel.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built FFN kernel: $(FFN_TARGET)"

ffn: dirs $(FFN_TARGET)
	@echo "Built FFN/MLP kernel (SwiGLU)"

# Embeddings kernel (token lookup + RoPE)
embeddings_kernel.o: embeddings_kernel.cu embeddings_kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(EMBEDDINGS_TARGET): embeddings_kernel.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built Embeddings kernel: $(EMBEDDINGS_TARGET)"

embeddings: dirs $(EMBEDDINGS_TARGET)
	@echo "Built Embeddings kernel (token lookup + RoPE)"

# Sampling kernel (temperature, top-k, top-p)
sampling_kernel.o: sampling_kernel.cu sampling_kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -lcurand

$(SAMPLING_TARGET): sampling_kernel.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart -lcurand
	@echo "Built Sampling kernel: $(SAMPLING_TARGET)"

sampling: dirs $(SAMPLING_TARGET)
	@echo "Built Sampling kernel (temperature, top-k, top-p)"

# Linear projection kernel (Q/K/V, output, LM head) - uses cuBLAS
linear_projection.o: linear_projection.cu linear_projection.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -lcublas

$(LINEAR_TARGET): linear_projection.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart -lcublas
	@echo "Built Linear projection kernel: $(LINEAR_TARGET)"

linear: dirs $(LINEAR_TARGET)
	@echo "Built Linear projection kernel (cuBLAS SGEMM)"

# Residual add kernel (element-wise add + fused RMSNorm)
residual_add.o: residual_add.cu residual_add.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(RESIDUAL_TARGET): residual_add.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built Residual add kernel: $(RESIDUAL_TARGET)"

residual: dirs $(RESIDUAL_TARGET)
	@echo "Built Residual add kernel (vectorized + fused RMSNorm)"

# Inference pipeline orchestrator
inference_pipeline.o: inference_pipeline.cu inference_pipeline.h inference_kernels.h linear_projection.h residual_add.h
	$(NVCC) $(INT8_NVCC_FLAGS) -c $< -o $@ -lcublas -lcurand

$(PIPELINE_TARGET): inference_pipeline.o flash_attention_int8.o rmsnorm_kernel.o ffn_kernel.o embeddings_kernel.o sampling_kernel.o linear_projection.o residual_add.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart -lcublas -lcurand
	@echo "Built Inference pipeline: $(PIPELINE_TARGET)"

pipeline: dirs $(PIPELINE_TARGET)
	@echo "Built complete inference pipeline orchestrator"

# Full inference pipeline library (all kernels in one .so)
INFERENCE_OBJS = flash_attention_int8.o rmsnorm_kernel.o ffn_kernel.o embeddings_kernel.o sampling_kernel.o linear_projection.o residual_add.o

$(INFERENCE_TARGET): $(INFERENCE_OBJS)
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart -lcurand -lcublas
	@echo "Built EdgeLLM Inference library: $(INFERENCE_TARGET)"

inference: dirs $(INFERENCE_OBJS) $(INFERENCE_TARGET)
	@echo "Built complete EdgeLLM inference pipeline"

# Build all inference kernels separately
inference-all: dirs rmsnorm ffn embeddings sampling linear residual int8
	@echo "Built all inference kernels (RMSNorm, FFN, Embeddings, Sampling, Linear, Residual, INT8 Attention)"

# Build unified inference library (recommended)
inference-unified: dirs $(INFERENCE_TARGET)
	@echo "Built unified EdgeLLM inference library"

# Build complete pipeline with orchestrator (FULL INFERENCE)
inference-pipeline: dirs $(PIPELINE_TARGET)
	@echo "Built complete EdgeLLM inference pipeline with orchestrator"

# =============================================================================
# cuBLAS Full GPU Inference (400+ tok/s target)
# =============================================================================
# This is the key kernel for achieving Ollama-level performance.
# Uses cuBLAS for all matmuls + proper GQA attention kernel.

cublas_matmul.o: cublas_matmul.cu cublas_matmul.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -lcublas

$(CUBLAS_TARGET): cublas_matmul.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart -lcublas
	@echo "Built cuBLAS matmul kernel: $(CUBLAS_TARGET)"

# Build cuBLAS kernel (RECOMMENDED for full GPU inference)
cublas: dirs cublas_matmul.o $(CUBLAS_TARGET)
	@echo "Built cuBLAS full GPU inference kernel (400+ tok/s target)"

# =============================================================================
# EAGLE Speculative Decoding (Tree Attention)
# =============================================================================
# Tree attention kernel for parallel verification of draft tokens
# Based on: https://arxiv.org/abs/2401.15077 (EAGLE)

TREE_ATTENTION_TARGET = $(LIB_DIR)/libtree_attention.$(SHARED_EXT)

tree_attention.o: tree_attention.cu tree_attention.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(TREE_ATTENTION_TARGET): tree_attention.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built Tree Attention kernel: $(TREE_ATTENTION_TARGET)"

# Build tree attention for speculative decoding
tree-attention: dirs tree_attention.o $(TREE_ATTENTION_TARGET)
	@echo "Built Tree Attention kernel (EAGLE speculative decoding)"

# =============================================================================
# EAGLE Speculative Decoding Full System
# =============================================================================
# Complete EAGLE implementation with:
# - eagle_inference.cu: Draft model forward, verification, generation loop
# - eagle_weights_init.cu: Weight initialization (random, identity-like, load/save)
# - edgellm_eagle.cu: Main test binary

EAGLE_TARGET = $(LIB_DIR)/libeagle_inference.$(SHARED_EXT)
EAGLE_BIN = $(BIN_DIR)/edgellm_eagle

eagle_inference.o: eagle_inference.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -lcublas -lcurand

eagle_weights_init.o: eagle_weights_init.cu eagle_weights_init.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -lcurand

$(EAGLE_TARGET): eagle_inference.o eagle_weights_init.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart -lcublas -lcurand
	@echo "Built EAGLE inference library: $(EAGLE_TARGET)"

# EAGLE test binary (requires cublas_matmul.o for INT4 inference)
edgellm_eagle.o: edgellm_eagle.cu eagle_weights_init.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(EAGLE_BIN): edgellm_eagle.o cublas_matmul.o eagle_inference.o eagle_weights_init.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lcudart -lcublas -lcurand
	@echo "Built EAGLE test binary: $(EAGLE_BIN)"

# Build EAGLE library
eagle-lib: dirs eagle_inference.o eagle_weights_init.o $(EAGLE_TARGET)
	@echo "Built EAGLE speculative decoding library"

# Build EAGLE test binary (RECOMMENDED for testing)
eagle: dirs cublas_matmul.o eagle_inference.o eagle_weights_init.o edgellm_eagle.o $(EAGLE_BIN)
	@echo "Built EAGLE speculative decoding test binary"
	@echo "Usage: $(EAGLE_BIN) -i 'prompt' -n 100 -d 6 --baseline"

# Build full EdgeLLM 2.0 (INT4 + EAGLE + Temperature Sampling)
edgellm2: dirs cublas_matmul.o eagle_inference.o eagle_weights_init.o edgellm_eagle.o $(EAGLE_BIN)
	@echo "Built EdgeLLM 2.0 with EAGLE speculative decoding"

# =============================================================================
# edge CLI - Minimal Ollama-style interface
# =============================================================================
# Usage: edge run qwen -p "What is 2+2?"

EDGE_BIN = $(BIN_DIR)/edge

edge_cli.o: edge_cli.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

sampling.o: sampling.cu sampling.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -lcurand

inference_with_sampling.o: inference_with_sampling.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ -lcurand

$(EDGE_BIN): edge_cli.o cublas_matmul.o sampling.o inference_with_sampling.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lcudart -lcublas -lcurand
	@echo ""
	@echo "Built: $(EDGE_BIN)"
	@echo ""
	@echo "Usage:"
	@echo "  edge run qwen              Interactive chat"
	@echo "  edge run qwen -p 'prompt'  Single generation"
	@echo "  edge models                List models"
	@echo ""

# Build edge CLI (RECOMMENDED for users)
edge: dirs cublas_matmul.o sampling.o inference_with_sampling.o edge_cli.o $(EDGE_BIN)
	@echo "Built edge CLI"

# =============================================================================
# Lookahead Decoding (Training-Free Speculative Decoding)
# =============================================================================
# Based on: https://arxiv.org/abs/2402.02057 (ICML 2024)
# Uses Jacobi iteration for parallel token prediction without draft model training

LOOKAHEAD_NGRAM_TARGET = $(LIB_DIR)/liblookahead_ngram.$(SHARED_EXT)
LOOKAHEAD_STATE_TARGET = $(LIB_DIR)/liblookahead_state.$(SHARED_EXT)
LOOKAHEAD_TARGET = $(LIB_DIR)/liblookahead.$(SHARED_EXT)
LOOKAHEAD_TEST_BIN = $(BIN_DIR)/test_lookahead

# N-gram pool for candidate caching
lookahead_ngram.o: lookahead_ngram.cu lookahead_ngram.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(LOOKAHEAD_NGRAM_TARGET): lookahead_ngram.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built Lookahead N-gram pool: $(LOOKAHEAD_NGRAM_TARGET)"

# Lookahead state management
lookahead_state.o: lookahead_state.cu lookahead_state.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(LOOKAHEAD_STATE_TARGET): lookahead_state.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built Lookahead state: $(LOOKAHEAD_STATE_TARGET)"

# Combined lookahead library
$(LOOKAHEAD_TARGET): lookahead_ngram.o lookahead_state.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built Lookahead decoding library: $(LOOKAHEAD_TARGET)"

# Build N-gram pool standalone
lookahead-ngram: dirs lookahead_ngram.o $(LOOKAHEAD_NGRAM_TARGET)
	@echo "Built Lookahead N-gram pool"

# Build state management standalone
lookahead-state: dirs lookahead_state.o $(LOOKAHEAD_STATE_TARGET)
	@echo "Built Lookahead state management"

# Build combined lookahead library (RECOMMENDED)
lookahead: dirs lookahead_ngram.o lookahead_state.o $(LOOKAHEAD_TARGET)
	@echo "Built Lookahead decoding library"

# Test binary for Phase 1 structures
$(LOOKAHEAD_TEST_BIN): test_lookahead.cu lookahead_ngram.o lookahead_state.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lcudart
	@echo "Built Lookahead test binary: $(LOOKAHEAD_TEST_BIN)"

# Run Lookahead tests
test-lookahead: dirs lookahead_ngram.o lookahead_state.o $(LOOKAHEAD_TEST_BIN)
	@echo "Running Lookahead decoding tests..."
	$(LOOKAHEAD_TEST_BIN)

# Lookahead Attention (Phase 2)
LOOKAHEAD_ATTN_TARGET = $(LIB_DIR)/liblookahead_attention.$(SHARED_EXT)
LOOKAHEAD_ATTN_TEST_BIN = $(BIN_DIR)/test_lookahead_attention

lookahead_attention.o: lookahead_attention.cu lookahead_attention.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(LOOKAHEAD_ATTN_TARGET): lookahead_attention.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built Lookahead attention: $(LOOKAHEAD_ATTN_TARGET)"

# Build lookahead attention library
lookahead-attention: dirs lookahead_attention.o $(LOOKAHEAD_ATTN_TARGET)
	@echo "Built Lookahead attention kernel"

# Test binary for lookahead attention
$(LOOKAHEAD_ATTN_TEST_BIN): test_lookahead_attention.cu lookahead_attention.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lcudart
	@echo "Built Lookahead attention test: $(LOOKAHEAD_ATTN_TEST_BIN)"

# Run lookahead attention tests
test-lookahead-attention: dirs lookahead_attention.o $(LOOKAHEAD_ATTN_TEST_BIN)
	@echo "Running Lookahead attention tests..."
	$(LOOKAHEAD_ATTN_TEST_BIN)

# Combined Lookahead library (Phase 1 + Phase 2)
LOOKAHEAD_FULL_TARGET = $(LIB_DIR)/liblookahead_full.$(SHARED_EXT)

$(LOOKAHEAD_FULL_TARGET): lookahead_ngram.o lookahead_state.o lookahead_attention.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built full Lookahead library: $(LOOKAHEAD_FULL_TARGET)"

# Build complete lookahead system (RECOMMENDED)
lookahead-full: dirs lookahead_ngram.o lookahead_state.o lookahead_attention.o $(LOOKAHEAD_FULL_TARGET)
	@echo "Built complete Lookahead decoding system (Phase 1 + 2)"

# Run all lookahead tests
test-lookahead-all: test-lookahead test-lookahead-attention
	@echo "All Lookahead tests completed"

# =============================================================================
# INT4 Quantized Inference (Target: 400+ tok/s)
# =============================================================================
# INT4 dequant + GEMV kernel for 8x memory bandwidth reduction
# Uses GROUP_SIZE=128 (AWQ-style) quantization

INT4_TARGET = $(LIB_DIR)/libint4_gemv.$(SHARED_EXT)

int4_gemv.o: int4_gemv.cu int4_gemv.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(INT4_TARGET): int4_gemv.o
	$(NVCC) $(SHARED_FLAGS) -o $@ $^ -lcudart
	@echo "Built INT4 GEMV kernel: $(INT4_TARGET)"

# Combined INT4 + cuBLAS (RECOMMENDED for production)
# Includes INT4 GEMV kernel + INT8 embedding kernel for fast logit computation
CUBLAS_INT4_TARGET = $(LIB_DIR)/libcublas_int4.$(SHARED_EXT)

cublas_int4.o: cublas_matmul.cu cublas_matmul.h int4_gemv.cu int4_gemv.h int8_embedding.h
	$(NVCC) $(NVCC_FLAGS) -c cublas_matmul.cu -o cublas_int4.o -lcublas

int4_gemv_standalone.o: int4_gemv.cu int4_gemv.h
	$(NVCC) $(NVCC_FLAGS) -c int4_gemv.cu -o int4_gemv_standalone.o

int8_embedding.o: int8_embedding.cu int8_embedding.h
	$(NVCC) $(NVCC_FLAGS) -c int8_embedding.cu -o int8_embedding.o

$(CUBLAS_INT4_TARGET): cublas_int4.o int4_gemv_standalone.o int8_embedding.o
	$(NVCC) $(SHARED_FLAGS) -o $@ cublas_int4.o int4_gemv_standalone.o int8_embedding.o -lcudart -lcublas
	@echo "Built cuBLAS + INT4 + INT8 embedding kernel: $(CUBLAS_INT4_TARGET)"

# Build INT4 kernel standalone
int4: dirs int4_gemv.o $(INT4_TARGET)
	@echo "Built INT4 GEMV kernel (standalone)"

# Build INT4 with cuBLAS integration (RECOMMENDED)
cublas-int4: dirs cublas_int4.o int4_gemv_standalone.o int8_embedding.o $(CUBLAS_INT4_TARGET)
	@echo "Built cuBLAS + INT4 + INT8 embedding (400+ tok/s target)"

# Build everything for full GPU inference
gpu-inference: dirs $(CUBLAS_TARGET)
	@echo "Built complete GPU inference library"

# Build full GPU inference with INT4 support
gpu-inference-int4: dirs $(CUBLAS_INT4_TARGET)
	@echo "Built complete GPU inference library with INT4 support"

# Test binary
test: dirs $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/test_cuda_kernel test_cuda_kernel.cu $(OBJECTS) -lcudart
	@echo "Running CUDA tests..."
	$(BIN_DIR)/test_cuda_kernel

# Flash Attention test
test-flash: dirs flash_attention.o
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/test_flash_attention test_flash_attention.cu flash_attention.o -lcudart
	@echo "Running Flash Attention tests..."
	$(BIN_DIR)/test_flash_attention

# INT8 Flash Attention test
test-int8: dirs flash_attention_int8.o
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/test_flash_attention_int8 test_flash_attention_int8.cu flash_attention_int8.o -lcudart
	@echo "Running INT8 Flash Attention tests..."
	$(BIN_DIR)/test_flash_attention_int8

# FlashAttention-2 accuracy test
test-fa2-accuracy: dirs flash_attention_v2.o
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/test_flash_attention_v2_accuracy test_flash_attention_v2_accuracy.cu flash_attention_v2.o -lcudart
	@echo "Running FlashAttention-2 accuracy tests..."
	$(BIN_DIR)/test_flash_attention_v2_accuracy

# Test all (single GPU)
test-all: test test-flash test-int8 test-fa2-accuracy

# Test all including multi-GPU
test-all-multi-gpu: test-all test-fa2-multi-gpu

# Clean
clean:
	rm -f $(OBJECTS) flash_attention_int8.o flash_attention_v2.o flash_attention_v2_multi_gpu.o flash_attention_v2_optimized.o $(TARGET) $(STATIC_TARGET) $(FA_TARGET) $(INT8_TARGET) $(FA2_TARGET) $(FA2_MG_TARGET) $(FA2_OPT_TARGET)
	rm -f rmsnorm_kernel.o ffn_kernel.o embeddings_kernel.o sampling_kernel.o linear_projection.o residual_add.o inference_pipeline.o
	rm -f $(RMSNORM_TARGET) $(FFN_TARGET) $(EMBEDDINGS_TARGET) $(SAMPLING_TARGET) $(LINEAR_TARGET) $(RESIDUAL_TARGET) $(PIPELINE_TARGET) $(INFERENCE_TARGET)
	rm -f cublas_matmul.o cublas_int4.o int4_gemv_standalone.o int8_embedding.o $(CUBLAS_TARGET) $(CUBLAS_INT4_TARGET) $(INT4_TARGET)
	rm -f tree_attention.o $(TREE_ATTENTION_TARGET)
	rm -f eagle_inference.o eagle_weights_init.o edgellm_eagle.o $(EAGLE_TARGET) $(EAGLE_BIN)
	rm -f lookahead_ngram.o lookahead_state.o lookahead_attention.o $(LOOKAHEAD_NGRAM_TARGET) $(LOOKAHEAD_STATE_TARGET) $(LOOKAHEAD_TARGET) $(LOOKAHEAD_TEST_BIN) $(LOOKAHEAD_ATTN_TARGET) $(LOOKAHEAD_ATTN_TEST_BIN) $(LOOKAHEAD_FULL_TARGET)
	rm -f edge_cli.o sampling.o inference_with_sampling.o $(EDGE_BIN)
	rm -f $(BIN_DIR)/test_cuda_kernel $(BIN_DIR)/test_flash_attention $(BIN_DIR)/test_flash_attention_int8 $(BIN_DIR)/test_flash_attention_v2_multi_gpu $(BIN_DIR)/test_flash_attention_v2_optimized $(BIN_DIR)/test_lookahead

# Install to system (optional)
install: $(TARGET)
	@echo "Installing to /usr/local/lib..."
	cp $(TARGET) /usr/local/lib/
	cp tmac_kernel_cuda.h /usr/local/include/

# Jetson Nano specific build
jetson-nano:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_53,code=sm_53"

# Jetson Orin specific build
jetson-orin:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_87,code=sm_87"

# RTX build (for development)
rtx:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89"

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: all

# Profiling build (for Nsight)
profile: NVCC_FLAGS += -lineinfo
profile: all

# Tensor Core build (INT8 support for sm_75+)
# Enables WMMA API for INT8 Tensor Core operations
tensorcore:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_75,code=sm_75 \
	                   -gencode arch=compute_80,code=sm_80 \
	                   -gencode arch=compute_86,code=sm_86 \
	                   -gencode arch=compute_87,code=sm_87 \
	                   -gencode arch=compute_89,code=sm_89" \
	        NVCC_FLAGS_COMMON="$(NVCC_FLAGS_COMMON) --expt-relaxed-constexpr"

# T4 specific build (Kaggle/Colab)
t4:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_75,code=sm_75" \
	        NVCC_FLAGS_COMMON="$(NVCC_FLAGS_COMMON) --expt-relaxed-constexpr"

# A100 specific build
a100:
	$(MAKE) CUDA_ARCH="-gencode arch=compute_80,code=sm_80" \
	        NVCC_FLAGS_COMMON="$(NVCC_FLAGS_COMMON) --expt-relaxed-constexpr"

# Print configuration
info:
	@echo "NVCC: $(shell which nvcc 2>/dev/null || echo 'not found')"
	@nvcc --version 2>/dev/null || echo "CUDA not available"
	@echo "Target: $(TARGET)"
	@echo "Flash Attention: $(FA_TARGET)"
	@echo "INT8 Flash Attention: $(INT8_TARGET)"
	@echo "FA2 Multi-GPU: $(FA2_MG_TARGET)"
	@echo "Inference Library: $(INFERENCE_TARGET)"
	@echo "cuBLAS Full GPU: $(CUBLAS_TARGET)"
	@echo "CUDA_ARCH: $(CUDA_ARCH)"
	@echo ""
	@echo "Build targets:"
	@echo "  make            - Default multi-arch build"
	@echo "  make flash      - Build with Flash Attention (FP32)"
	@echo "  make int8       - Build INT8 Tensor Core Flash Attention"
	@echo "  make fa2        - Build FlashAttention-2 (tiled)"
	@echo "  make fa2-opt    - Build FlashAttention-2 Optimized (WMMA+GPU-Resident+Split-K)"
	@echo "  make fa2-multi-gpu - Build Multi-GPU FlashAttention-2 (NCCL)"
	@echo "  make all-int8   - Build all including INT8"
	@echo "  make all-attention - Build all attention variants"
	@echo "  make all-attention-multi-gpu - Build all including multi-GPU"
	@echo "  make tensorcore - INT8 Tensor Core optimized (sm_75+)"
	@echo "  make t4         - Tesla T4 optimized (Kaggle/Colab)"
	@echo "  make a100       - A100 optimized"
	@echo "  make jetson-nano - Jetson Nano (sm_53)"
	@echo "  make jetson-orin - Jetson Orin (sm_87)"
	@echo "  make rtx        - RTX 30/40 series"
	@echo ""
	@echo "Full GPU inference (80+ tok/s):"
	@echo "  make edge       - Build 'edge' CLI (RECOMMENDED for users)"
	@echo "  make cublas     - Build cuBLAS matmul + GQA attention"
	@echo "  make gpu-inference - Build complete GPU inference library"
	@echo ""
	@echo "Inference pipeline targets:"
	@echo "  make rmsnorm    - Build RMSNorm kernel"
	@echo "  make ffn        - Build FFN/MLP kernel (SwiGLU)"
	@echo "  make embeddings - Build Embeddings kernel (token lookup + RoPE)"
	@echo "  make sampling   - Build Sampling kernel (temperature, top-k, top-p)"
	@echo "  make linear     - Build Linear projection kernel (Q/K/V, LM head)"
	@echo "  make residual   - Build Residual add kernel (+ fused RMSNorm)"
	@echo "  make pipeline   - Build inference pipeline orchestrator"
	@echo "  make inference-all      - Build all inference kernels separately"
	@echo "  make inference-unified  - Build unified inference library"
	@echo "  make inference-pipeline - Build complete pipeline"
	@echo ""
	@echo "Speculative decoding targets:"
	@echo "  make eagle      - Build EAGLE speculative decoding (requires draft model)"
	@echo "  make lookahead  - Build Lookahead decoding Phase 1 (training-free)"
	@echo "  make lookahead-attention - Build Lookahead attention Phase 2"
	@echo "  make lookahead-full - Build complete Lookahead system (RECOMMENDED)"
	@echo "  make lookahead-ngram - Build N-gram pool only"
	@echo "  make lookahead-state - Build state management only"
	@echo ""
	@echo "Test targets:"
	@echo "  make test       - Run T-MAC kernel tests"
	@echo "  make test-flash - Run Flash Attention tests (FP32)"
	@echo "  make test-int8  - Run INT8 Flash Attention tests"
	@echo "  make test-fa2-accuracy - Run FA2 accuracy tests"
	@echo "  make test-fa2-opt      - Run Optimized FA2 tests"
	@echo "  make test-fa2-multi-gpu - Run Multi-GPU FA2 tests"
	@echo "  make test-lookahead    - Run Lookahead Phase 1 tests"
	@echo "  make test-lookahead-attention - Run Lookahead attention tests"
	@echo "  make test-lookahead-all - Run all Lookahead tests"
	@echo "  make test-all   - Run all tests"
	@echo "  make test-all-multi-gpu - Run all tests including multi-GPU"
