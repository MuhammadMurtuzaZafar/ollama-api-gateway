#!/bin/bash
# EdgeLLM Benchmark Runner
# Runs comprehensive benchmark comparison in Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "EdgeLLM Benchmark Runner"
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

# Build benchmark Docker image
echo "Building benchmark Docker image..."
docker build -f "$PROJECT_DIR/Dockerfile.benchmark" -t edgellm-benchmark "$PROJECT_DIR"

# Run EdgeLLM benchmark in Docker
echo ""
echo "========================================"
echo "Running EdgeLLM Benchmark (Docker)"
echo "========================================"
docker run --rm \
    -v "$RESULTS_DIR:/workspace/results" \
    edgellm-benchmark \
    python3 benchmarks/edgellm_benchmark.py \
        --backend edgellm \
        --model models/smollm-135m.tmac2.bin \
        --runs 100 \
        --output /workspace/results/edgellm_${TIMESTAMP}.json

# Check if Ollama is running locally
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "========================================"
    echo "Running Ollama Benchmark (Local)"
    echo "========================================"

    # Pull model if needed
    ollama pull smollm:135m 2>/dev/null || true

    # Run Ollama benchmark (native, not in Docker)
    python3 "$PROJECT_DIR/benchmarks/edgellm_benchmark.py" \
        --backend ollama \
        --model smollm:135m \
        --runs 100 \
        --output "$RESULTS_DIR/ollama_${TIMESTAMP}.json"

    # Run comparison
    echo ""
    echo "========================================"
    echo "Running Comparison Benchmark"
    echo "========================================"
    python3 "$PROJECT_DIR/benchmarks/edgellm_benchmark.py" \
        --compare \
        --edgellm-model models/smollm-135m.tmac2.bin \
        --ollama-model smollm:135m \
        --runs 50 \
        --output "$RESULTS_DIR/comparison_${TIMESTAMP}.json"
else
    echo ""
    echo "Ollama not running. Skipping Ollama benchmark."
    echo "Start Ollama with: ollama serve"
fi

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo "Results saved to: $RESULTS_DIR"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "No JSON results found"
