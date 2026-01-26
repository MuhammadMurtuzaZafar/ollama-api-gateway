# Large LLM Benchmark Plan: EdgeLLM vs Ollama

## Objective

Compare EdgeLLM's FlashAttention-2 CUDA kernels against Ollama across a range of model sizes to:
1. Validate performance scaling with model size
2. Identify optimal model sizes for edge deployment
3. Generate paper-ready benchmark data

## Target Models

### Tier 1: Small Models (Edge-Deployable)
| Model | Parameters | Ollama Name | Expected VRAM | Priority |
|-------|------------|-------------|---------------|----------|
| SmolLM-135M | 135M | `smollm:135m` | 0.5 GB | Done |
| Qwen2.5-0.5B | 500M | `qwen2.5:0.5b` | 1 GB | High |
| Qwen2.5-1.5B | 1.5B | `qwen2.5:1.5b` | 3 GB | High |

### Tier 2: Medium Models (Jetson/RTX)
| Model | Parameters | Ollama Name | Expected VRAM | Priority |
|-------|------------|-------------|---------------|----------|
| Llama3.2-1B | 1B | `llama3.2:1b` | 2 GB | High |
| Llama3.2-3B | 3B | `llama3.2:3b` | 6 GB | High |
| Phi-3.5-mini | 3.8B | `phi3.5:3.8b` | 7 GB | Medium |
| Gemma2-2B | 2B | `gemma2:2b` | 4 GB | Medium |

### Tier 3: Large Models (Desktop GPU)
| Model | Parameters | Ollama Name | Expected VRAM | Priority |
|-------|------------|-------------|---------------|----------|
| Llama3.1-8B | 8B | `llama3.1:8b` | 16 GB | Medium |
| Qwen2.5-7B | 7B | `qwen2.5:7b` | 14 GB | Medium |
| Mistral-7B | 7B | `mistral:7b` | 14 GB | Low |

## Hardware Requirements

### Local Testing (Available)
- **MacBook Pro M1/M2**: CPU-only testing (Metal support planned)
- **Docker with CUDA**: For GPU testing on Linux

### Cloud GPU Testing
| Platform | GPU | VRAM | Cost | Best For |
|----------|-----|------|------|----------|
| Kaggle | T4 x2 | 15GB x2 | Free | Tier 1-2 models |
| Colab Pro | A100 | 40GB | $10/month | Tier 2-3 models |
| Lambda Labs | A100 | 40GB | $1.10/hr | Full benchmark |
| RunPod | RTX 4090 | 24GB | $0.44/hr | Tier 1-3 models |

## Benchmark Methodology

### 1. Standard Test Parameters
```bash
# Common settings for all benchmarks
WARMUP_RUNS=20
BENCHMARK_RUNS=100
TOKENS_PER_RUN=128
TEMPERATURE=0.0  # Deterministic
```

### 2. Test Prompts (Consistent Across All Models)
```python
PROMPTS = [
    "Hello",  # Minimal
    "What is machine learning?",  # Short
    "Explain quantum computing in simple terms.",  # Medium
    "Write a detailed explanation of how neural networks work, including backpropagation.",  # Long
]
```

### 3. Metrics to Capture
- **Throughput**: tokens/second (mean, std, min, max)
- **Latency**: P50, P95, P99, jitter (std)
- **TTFT**: Time to first token
- **Memory**: Peak GPU memory usage
- **Power**: GPU power draw (if available)

## Implementation Plan

### Phase 1: Infrastructure Setup (Day 1)
- [ ] Update benchmark script to support multiple models
- [ ] Add GPU memory monitoring
- [ ] Create automated Colab/Kaggle notebooks
- [ ] Set up results aggregation

### Phase 2: Small Model Benchmarks (Day 1-2)
- [ ] SmolLM-135M (baseline - already done: 708 tok/s)
- [ ] Qwen2.5-0.5B
- [ ] Qwen2.5-1.5B
- [ ] Llama3.2-1B

### Phase 3: Medium Model Benchmarks (Day 2-3)
- [ ] Llama3.2-3B
- [ ] Phi-3.5-mini (3.8B)
- [ ] Gemma2-2B

### Phase 4: Large Model Benchmarks (Day 3-4)
- [ ] Llama3.1-8B (requires A100 or RTX 4090)
- [ ] Qwen2.5-7B
- [ ] Mistral-7B

### Phase 5: Analysis & Report (Day 4-5)
- [ ] Generate comparison charts
- [ ] Calculate scaling efficiency
- [ ] Write benchmark report
- [ ] Update paper results

## Expected Results

Based on FlashAttention-2 performance (708 tok/s @ SmolLM-135M on T4):

| Model | Params | Ollama (est.) | EdgeLLM FA2 (est.) | Speedup |
|-------|--------|---------------|---------------------|---------|
| SmolLM-135M | 135M | 423 tok/s | **708 tok/s** | 1.67x |
| Qwen2.5-0.5B | 500M | ~300 tok/s | ~550 tok/s | 1.8x |
| Llama3.2-1B | 1B | ~200 tok/s | ~400 tok/s | 2.0x |
| Llama3.2-3B | 3B | ~100 tok/s | ~180 tok/s | 1.8x |
| Llama3.1-8B | 8B | ~50 tok/s | ~90 tok/s | 1.8x |

*Note: FA2 advantage increases with sequence length due to O(N) vs O(N^2) memory*

## Commands

### Pull All Ollama Models
```bash
# Tier 1
ollama pull smollm:135m
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:1.5b

# Tier 2
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull phi3.5:3.8b
ollama pull gemma2:2b

# Tier 3
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

### Run Benchmarks
```bash
# Single model
python benchmarks/edgellm_benchmark.py --backend ollama --model qwen2.5:0.5b --runs 100

# Comparison
python benchmarks/edgellm_benchmark.py --compare --ollama-model qwen2.5:1.5b --runs 100 -o results/qwen2.5-1.5b.json
```

### Automated Full Suite
```bash
# Run all benchmarks (to be implemented)
python benchmarks/run_all_benchmarks.py --output-dir results/
```

## Kaggle/Colab Notebook Structure

```python
# Cell 1: Setup
!nvidia-smi
!apt-get install -y libnccl2 libnccl-dev

# Cell 2: Clone and Build
!git clone --depth 1 https://github.com/umerkhan95/ollama-api-gateway.git
%cd ollama-api-gateway/mojo-gateway/src/kernels/cuda
!make CUDA_ARCH="-gencode arch=compute_75,code=sm_75" fa2

# Cell 3: Install Ollama
!curl -fsSL https://ollama.ai/install.sh | sh
!ollama serve &
!sleep 5
!ollama pull qwen2.5:0.5b

# Cell 4: Run FA2 Benchmark
!./bin/test_flash_attention_v2_accuracy

# Cell 5: Run Ollama Benchmark
!python benchmarks/edgellm_benchmark.py --backend ollama --model qwen2.5:0.5b --runs 50

# Cell 6: Compare Results
# Parse and display comparison
```

## Success Criteria

1. **Performance**: EdgeLLM FA2 >= 1.5x faster than Ollama on all models
2. **Accuracy**: Max error < 1e-5 on all model sizes
3. **Scaling**: Near-linear throughput scaling with model size
4. **Reproducibility**: < 5% variance across runs

## Next Steps

1. Start with Qwen2.5-0.5B benchmark (smallest step up from SmolLM)
2. Create automated benchmark runner script
3. Set up cloud GPU instance for larger models
4. Generate paper-quality comparison charts
