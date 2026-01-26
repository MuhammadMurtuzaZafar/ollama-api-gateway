#!/bin/bash
# Comprehensive benchmark of all Mojo LLM implementations

echo "============================================================"
echo "Mojo LLM Inference Benchmark Suite"
echo "============================================================"
echo ""

# Number of tokens to generate
TOKENS=128
TEMP=0.0

echo "Settings: $TOKENS tokens, temperature=$TEMP"
echo ""

# Build all implementations
echo "Building all implementations with -O3..."
mojo build -O3 src/llama2.mojo -o /tmp/llama2_basic 2>/dev/null
mojo build -O3 src/llama2_simd.mojo -o /tmp/llama2_simd 2>/dev/null
mojo build -O3 src/llama2_parallel.mojo -o /tmp/llama2_parallel 2>/dev/null
mojo build -O3 src/llama2_int4.mojo -o /tmp/llama2_int4 2>/dev/null
echo "Build complete."
echo ""

echo "============================================================"
echo "1. BASIC IMPLEMENTATION (List-based, no SIMD)"
echo "============================================================"
/tmp/llama2_basic src/stories110M.bin -z src/tokenizer.bin -n $TOKENS -t $TEMP 2>&1 | tail -5
echo ""

echo "============================================================"
echo "2. SIMD IMPLEMENTATION (UnsafePointer + SIMD vectorization)"
echo "============================================================"
/tmp/llama2_simd src/stories110M.bin -z src/tokenizer.bin -n $TOKENS -t $TEMP 2>&1 | tail -5
echo ""

echo "============================================================"
echo "3. PARALLEL + SIMD (Multi-threaded matmul)"
echo "============================================================"
/tmp/llama2_parallel src/stories110M.bin -z src/tokenizer.bin -n $TOKENS -t $TEMP 2>&1 | tail -5
echo ""

echo "============================================================"
echo "4. INT4 QUANTIZED (7x memory reduction)"
echo "============================================================"
/tmp/llama2_int4 src/stories110M.q4.bin -z src/tokenizer.bin -n $TOKENS -t $TEMP 2>&1 | tail -5
echo ""

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Model: stories110M (110M parameters)"
echo ""
echo "| Implementation     | Model Size | Speed       |"
echo "|--------------------|------------|-------------|"
