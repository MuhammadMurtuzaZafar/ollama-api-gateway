# EdgeLLM

> **Fine-tune once, deploy everywhere — from cloud to $15 edge devices**

High-performance LLM inference engine with **2.5x faster GPU attention** and **15.5x lower latency jitter** than Ollama. Built with Mojo for deterministic real-time performance.

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/umerkhan95/EdgeLLM)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Mojo](https://img.shields.io/badge/mojo-0.25.7-orange.svg)](https://www.modular.com/mojo)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://hub.docker.com)

---

## Benchmarks

### GPU Performance (Tesla T4)

| Metric | Ollama | EdgeLLM | Winner |
|--------|--------|---------|--------|
| **Attention Throughput** | ~598 tok/s | **1,490 tok/s** | EdgeLLM **2.5x** |
| **Layer Latency** | N/A | 27.97 μs | EdgeLLM |

### CPU Performance (x86)

| Metric | Ollama | EdgeLLM | Winner |
|--------|--------|---------|--------|
| **Latency Jitter** | 5,799 ms | **373 ms** | EdgeLLM **15.5x** |
| **Model Size** | 91 MB | **40 MB** | EdgeLLM **2.3x** |

---

## Quick Start

### Install (One-Liner)

```bash
curl -fsSL https://raw.githubusercontent.com/umerkhan95/EdgeLLM/main/mojo-gateway/install.sh | bash
```

### Install via Pixi

```bash
pixi add edgellm --channel https://prefix.dev/edgellm
```

### Usage

```bash
# List available models
edgellm models

# Download a model
edgellm pull smollm-135m

# Interactive chat
edgellm run smollm-135m

# Start API server
edgellm serve smollm-135m --port 8080
```

---

## Features

### Inference Engine
- **BitNet 1.58-bit** quantization (4.8x compression)
- **T-MAC** lookup table inference (no multiplication)
- **INT8 `__dp4a`** GPU kernels (Turing+)
- **Zero-copy** KV cache management
- **Deterministic latency** (no GC pauses)

### API Gateway
- **Multi-tenant** API key management
- **Role-based** access control (admin/user)
- **Rate limiting** per API key
- **Usage analytics** and monitoring
- **PostgreSQL** persistent storage

### Frontend Dashboard
- **React + Vite** modern UI
- **Dark mode** support
- **Interactive playground**
- **Real-time statistics**

---

## CUDA Kernel Status

Complete GPU inference pipeline for LLaMA-style transformers.

| Kernel | Status | Features |
|--------|--------|----------|
| **Attention (INT8)** | ✅ Complete | `__dp4a` Flash Attention, 2.5x faster than Ollama |
| **RMSNorm** | ✅ Complete | Warp reductions, vectorized, fused residual |
| **FFN/MLP** | ✅ Complete | SwiGLU activation, tiled, INT8 quantized |
| **Embeddings** | ✅ Complete | Token lookup, RoPE positional encoding |
| **Sampling** | ✅ Complete | Temperature, Top-K, Top-P, greedy |

### Build CUDA Kernels

```bash
cd mojo-gateway/src/kernels/cuda

# Build unified inference library (recommended)
make inference-unified

# Or build individual kernels
make rmsnorm ffn embeddings sampling int8

# Platform-specific builds
make t4          # Tesla T4 (Kaggle/Colab)
make jetson-nano # Jetson Nano
make rtx         # RTX 30/40 series
```

### Kernel Files

| File | Purpose |
|------|---------|
| `flash_attention_int8.cu` | INT8 dp4a attention (2.5x faster) |
| `rmsnorm_kernel.cu` | RMS Layer Normalization |
| `ffn_kernel.cu` | Feed-Forward Network (SwiGLU) |
| `embeddings_kernel.cu` | Token embeddings + RoPE |
| `sampling_kernel.cu` | Sampling strategies |
| `inference_kernels.h` | Unified header |

---

## Architecture

```
EdgeLLM/
├── mojo-gateway/              # Mojo inference engine
│   ├── src/
│   │   ├── edgellm_cli.mojo   # Ollama-style CLI
│   │   ├── bitnet_tmac_lut.mojo
│   │   └── kernels/
│   │       └── cuda/          # INT8 dp4a kernels
│   ├── install.sh             # One-liner installer
│   └── conda-recipe/          # Pixi/Magic distribution
│
├── backend/                   # FastAPI gateway
│   ├── main.py                # API endpoints
│   └── database.py            # PostgreSQL models
│
└── frontend/                  # React dashboard
    └── src/
        ├── pages/
        └── components/
```

---

## Full Stack Deployment

### Docker Compose

```bash
cd mojo-gateway
docker compose -f docker-compose.fullstack.yml up -d
```

**Services:**
- **API Gateway**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000
- **EdgeLLM**: http://localhost:8080

### Demo API Keys

| Role | Key | Rate Limit |
|------|-----|------------|
| Admin | `edgellm-admin-demo-key-12345` | 1000/hr |
| User | `edgellm-user-demo-key-67890` | 100/hr |

---

## API Usage

### Chat Completion

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer edgellm-user-demo-key-67890" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smollm-135m",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/chat",
    headers={"Authorization": "Bearer edgellm-user-demo-key-67890"},
    json={
        "model": "smollm-135m",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
print(response.json())
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer edgellm-user-demo-key-67890',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: 'smollm-135m',
    messages: [{ role: 'user', content: 'Hello!' }]
  })
});
```

---

## Supported Models

| Model | Parameters | Size | Use Case |
|-------|------------|------|----------|
| `smollm-135m` | 135M | 40 MB | Edge devices, IoT |
| `qwen2-0.5b` | 500M | 156 MB | General chat |
| `llama-3.2-1b` | 1B | 312 MB | Complex tasks |
| `phi-3-mini` | 3.8B | 1.2 GB | High quality |

---

## Hardware Support

### GPU (CUDA)

| Device | GPU | Expected Speed |
|--------|-----|----------------|
| Jetson Nano | Maxwell 128 | 80-120 tok/s |
| Tesla T4 | Turing 2560 | **1,490 tok/s** |
| RTX 3090 | Ampere 10496 | 400-600 tok/s |
| RTX 4090 | Ada 16384 | 600-1000 tok/s |

### CPU (Edge)

| Device | Price | Expected Speed |
|--------|-------|----------------|
| Pi Zero 2 W | **$15** | 5-10 tok/s |
| Pi 4 | $35 | 8-15 tok/s |
| Pi 5 | $80 | 20-40 tok/s |

---

## Platform Support

| Platform | Native | Docker |
|----------|--------|--------|
| Linux x86_64 | ✅ | ✅ |
| Linux ARM64 | ✅ | ✅ |
| macOS ARM64 | ✅ | ✅ |
| macOS x86_64 | ❌ | ✅ |
| Windows | ❌ | ✅ (WSL2) |

---

## Development

### Prerequisites

- [Pixi](https://pixi.sh) - Mojo package manager
- [Docker](https://docker.com) - Container runtime
- [Node.js 18+](https://nodejs.org) - Frontend

### Build from Source

```bash
# Clone
git clone https://github.com/umerkhan95/EdgeLLM.git
cd EdgeLLM/mojo-gateway

# Install dependencies
pixi install

# Build CLI
pixi run build-cli

# Run
./bin/edgellm --help
```

### Run Tests

```bash
# API tests (Jupyter notebook)
jupyter notebook notebooks/test_edgellm_api.ipynb

# Benchmark
python benchmarks/edgellm_benchmark.py --compare
```

---

## Key Technologies

- **[Mojo](https://www.modular.com/mojo)** - Systems language (no GC)
- **[BitNet](https://arxiv.org/abs/2402.17764)** - 1.58-bit quantization
- **[T-MAC](https://arxiv.org/abs/2407.00088)** - Lookup table inference
- **[FastAPI](https://fastapi.tiangolo.com)** - Async Python API
- **[React](https://react.dev)** - Frontend UI

---

## Documentation

- [Installation Guide](mojo-gateway/INSTALL.md)
- [Benchmark Report](mojo-gateway/BENCHMARK_REPORT.md)
- [API Documentation](http://localhost:8000/docs)
- [Edge Device Guide](mojo-gateway/EDGE_DEVICE_GUIDE.md)

---

## Contributing

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/EdgeLLM.git

# Create branch
git checkout -b feature/amazing-feature

# Make changes and commit
git commit -m "Add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Links

- **GitHub**: https://github.com/umerkhan95/EdgeLLM
- **Issues**: https://github.com/umerkhan95/EdgeLLM/issues
- **Discussions**: https://github.com/umerkhan95/EdgeLLM/discussions

---

<p align="center">
  <strong>EdgeLLM</strong> — LLM inference for the real world
</p>
