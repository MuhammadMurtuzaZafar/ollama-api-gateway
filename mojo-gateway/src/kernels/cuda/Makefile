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

SOURCES = tmac_kernel.cu flash_attention.cu
OBJECTS = tmac_kernel.o flash_attention.o
HEADERS = tmac_kernel_cuda.h flash_attention.h flash_attention_int8.h

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

# INT8 Tensor Core Flash Attention (requires sm_75+)
flash_attention_int8.o: flash_attention_int8.cu flash_attention_int8.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

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
all-attention: dirs $(FA_TARGET) $(INT8_TARGET) $(FA2_TARGET)
	@echo "Built all attention kernels (FP32, INT8, FA2)"

# Build all attention variants including multi-GPU
all-attention-multi-gpu: dirs $(FA_TARGET) $(INT8_TARGET) $(FA2_TARGET) $(FA2_MG_TARGET)
	@echo "Built all attention kernels (FP32, INT8, FA2, Multi-GPU)"

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
	rm -f $(OBJECTS) flash_attention_int8.o flash_attention_v2.o flash_attention_v2_multi_gpu.o $(TARGET) $(STATIC_TARGET) $(FA_TARGET) $(INT8_TARGET) $(FA2_TARGET) $(FA2_MG_TARGET)
	rm -f $(BIN_DIR)/test_cuda_kernel $(BIN_DIR)/test_flash_attention $(BIN_DIR)/test_flash_attention_int8 $(BIN_DIR)/test_flash_attention_v2_multi_gpu

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
	@echo "CUDA_ARCH: $(CUDA_ARCH)"
	@echo ""
	@echo "Build targets:"
	@echo "  make            - Default multi-arch build"
	@echo "  make flash      - Build with Flash Attention (FP32)"
	@echo "  make int8       - Build INT8 Tensor Core Flash Attention"
	@echo "  make fa2        - Build FlashAttention-2 (tiled)"
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
	@echo "Test targets:"
	@echo "  make test       - Run T-MAC kernel tests"
	@echo "  make test-flash - Run Flash Attention tests (FP32)"
	@echo "  make test-int8  - Run INT8 Flash Attention tests"
	@echo "  make test-fa2-multi-gpu - Run Multi-GPU FA2 tests"
	@echo "  make test-all   - Run all tests"
	@echo "  make test-all-multi-gpu - Run all tests including multi-GPU"
