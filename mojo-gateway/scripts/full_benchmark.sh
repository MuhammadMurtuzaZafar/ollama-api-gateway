#!/bin/bash
# EdgeLLM Full Benchmark Suite
# Builds Docker, runs real inference, collects quality metrics, generates report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "EdgeLLM Full Benchmark Suite"
echo "========================================"
echo "Project: $PROJECT_DIR"
echo "Results: $RESULTS_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# ==============================================================================
# Phase 1: Build Docker Image with Mojo
# ==============================================================================
echo ""
echo "========================================"
echo "Phase 1: Building Docker Image"
echo "========================================"

cd "$PROJECT_DIR"
docker build -f Dockerfile.mojo -t edgellm-inference . 2>&1 | tee "$RESULTS_DIR/docker_build_${TIMESTAMP}.log"

# ==============================================================================
# Phase 2: Run EdgeLLM Inference Test in Docker
# ==============================================================================
echo ""
echo "========================================"
echo "Phase 2: Testing EdgeLLM Inference"
echo "========================================"

# Test if binary was built successfully
docker run --rm edgellm-inference ls -la /workspace/bin/ 2>&1 || true

# Run inference test
echo "Running EdgeLLM inference test..."
docker run --rm \
    -v "$RESULTS_DIR:/workspace/results" \
    edgellm-inference \
    bash -c '
        if [ -f /workspace/bin/edgellm ]; then
            echo "EdgeLLM binary found!"
            /workspace/bin/edgellm /workspace/models/smollm-135m.tmac2.bin -n 10 -t 0 2>&1 || echo "Inference test completed (check output above)"
        else
            echo "EdgeLLM binary not found. Checking Mojo build..."
            pixi run mojo --version
            echo "Attempting to build..."
            pixi run mojo build -O3 /workspace/src/bitnet_tmac_lut.mojo -o /workspace/bin/edgellm 2>&1
        fi
    ' 2>&1 | tee "$RESULTS_DIR/inference_test_${TIMESTAMP}.log"

# ==============================================================================
# Phase 3: Run Performance Benchmark in Docker
# ==============================================================================
echo ""
echo "========================================"
echo "Phase 3: Running Performance Benchmark"
echo "========================================"

docker run --rm \
    -v "$RESULTS_DIR:/workspace/results" \
    edgellm-inference \
    python3 /workspace/benchmarks/edgellm_benchmark.py \
        --backend edgellm \
        --model /workspace/models/smollm-135m.tmac2.bin \
        --runs 50 \
        --output /workspace/results/edgellm_perf_${TIMESTAMP}.json \
    2>&1 | tee -a "$RESULTS_DIR/benchmark_${TIMESTAMP}.log"

# ==============================================================================
# Phase 4: Run Ollama Benchmark (if available)
# ==============================================================================
echo ""
echo "========================================"
echo "Phase 4: Running Ollama Benchmark"
echo "========================================"

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is running, starting benchmark..."

    # Ensure model is available
    ollama pull smollm:135m 2>/dev/null || true

    python3 "$PROJECT_DIR/benchmarks/edgellm_benchmark.py" \
        --backend ollama \
        --model smollm:135m \
        --runs 50 \
        --output "$RESULTS_DIR/ollama_perf_${TIMESTAMP}.json" \
    2>&1 | tee -a "$RESULTS_DIR/benchmark_${TIMESTAMP}.log"
else
    echo "Ollama not running. Skipping Ollama benchmark."
    echo "Start Ollama with: ollama serve"
fi

# ==============================================================================
# Phase 5: Run Quality Metrics
# ==============================================================================
echo ""
echo "========================================"
echo "Phase 5: Running Quality Metrics"
echo "========================================"

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    python3 "$PROJECT_DIR/benchmarks/quality_metrics.py" \
        --ollama-model smollm:135m \
        --output "$RESULTS_DIR/quality_${TIMESTAMP}.json" \
    2>&1 | tee -a "$RESULTS_DIR/quality_${TIMESTAMP}.log"
else
    echo "Ollama not running. Quality metrics skipped."
fi

# ==============================================================================
# Phase 6: Run Comparison Benchmark
# ==============================================================================
echo ""
echo "========================================"
echo "Phase 6: Running Comparison Benchmark"
echo "========================================"

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    python3 "$PROJECT_DIR/benchmarks/edgellm_benchmark.py" \
        --compare \
        --edgellm-model models/smollm-135m.tmac2.bin \
        --ollama-model smollm:135m \
        --runs 100 \
        --output "$RESULTS_DIR/comparison_${TIMESTAMP}.json" \
    2>&1 | tee -a "$RESULTS_DIR/benchmark_${TIMESTAMP}.log"
fi

# ==============================================================================
# Phase 7: Generate Final Report
# ==============================================================================
echo ""
echo "========================================"
echo "Phase 7: Generating Final Report"
echo "========================================"

# Create summary report
cat > "$RESULTS_DIR/BENCHMARK_SUMMARY_${TIMESTAMP}.md" << 'REPORT_EOF'
# EdgeLLM Benchmark Summary

**Generated:** TIMESTAMP_PLACEHOLDER
**Platform:** Docker (Ubuntu 22.04) + Host macOS

## Files Generated

| File | Description |
|------|-------------|
| `docker_build_*.log` | Docker build output |
| `inference_test_*.log` | EdgeLLM inference test |
| `edgellm_perf_*.json` | EdgeLLM performance metrics |
| `ollama_perf_*.json` | Ollama performance metrics |
| `quality_*.json` | Output quality comparison |
| `comparison_*.json` | Full comparison results |

## Quick Summary

See individual JSON files for detailed metrics.

### Key Metrics to Check

1. **Throughput (tok/s)**: Higher is better
2. **Latency Jitter (ms)**: Lower is better - EdgeLLM's key advantage
3. **P99 Latency (ms)**: Lower is better for real-time
4. **Quality (keyword match)**: Should be comparable between systems

REPORT_EOF

# Replace timestamp
sed -i.bak "s/TIMESTAMP_PLACEHOLDER/$TIMESTAMP/g" "$RESULTS_DIR/BENCHMARK_SUMMARY_${TIMESTAMP}.md" 2>/dev/null || \
    sed "s/TIMESTAMP_PLACEHOLDER/$TIMESTAMP/g" "$RESULTS_DIR/BENCHMARK_SUMMARY_${TIMESTAMP}.md" > "$RESULTS_DIR/temp.md" && \
    mv "$RESULTS_DIR/temp.md" "$RESULTS_DIR/BENCHMARK_SUMMARY_${TIMESTAMP}.md"

# ==============================================================================
# Done!
# ==============================================================================
echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
ls -la "$RESULTS_DIR"/*${TIMESTAMP}* 2>/dev/null || ls -la "$RESULTS_DIR"
echo ""
echo "To view comparison results:"
echo "  cat $RESULTS_DIR/comparison_${TIMESTAMP}.json | jq '.results'"
