# EdgeLLM Installation Guide

## Quick Install (One-Liner)

```bash
curl -fsSL https://raw.githubusercontent.com/umerkhan95/EdgeLLM/main/mojo-gateway/install.sh | bash
```

## Install via Pixi/Magic

```bash
# Add the EdgeLLM channel
pixi project channel add https://prefix.dev/edgellm

# Install EdgeLLM
pixi add edgellm
```

Or in a new project:

```bash
pixi init my-llm-project
cd my-llm-project
pixi add edgellm --channel https://prefix.dev/edgellm
pixi run edgellm --help
```

## Manual Installation

### Prerequisites

- **Pixi** - Mojo package manager
  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/umerkhan95/EdgeLLM.git
cd ollama-api-gateway/mojo-gateway

# 2. Install dependencies
pixi install

# 3. Build the CLI
pixi run build-cli

# 4. Run EdgeLLM
./bin/edgellm --help
```

## Docker Installation

For platforms without native Mojo support (e.g., macOS x86_64):

```bash
cd mojo-gateway
docker compose -f docker-compose.mojo.yml up --build
```

## Platform Support

| Platform | Native | Docker | Notes |
|----------|--------|--------|-------|
| Linux x86_64 | Yes | Yes | Full support |
| Linux ARM64 | Yes | Yes | Jetson, Pi 4+ |
| macOS ARM64 | Yes | Yes | Apple Silicon |
| macOS x86_64 | No | Yes | Docker only |
| Windows | No | Yes | WSL2 + Docker |

## Verify Installation

```bash
# Check version
edgellm --help

# List available models
edgellm models

# Download a model
edgellm pull smollm-135m

# Run interactive chat
edgellm run smollm-135m

# Start API server
edgellm serve smollm-135m --port 8080
```

## Uninstall

```bash
# Using install script
curl -fsSL https://raw.githubusercontent.com/umerkhan95/EdgeLLM/main/mojo-gateway/install.sh | bash -s uninstall

# Or manually
rm -rf ~/.edgellm
rm -f ~/.local/bin/edgellm
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGELLM_HOME` | `~/.edgellm` | Installation directory |
| `EDGELLM_MODELS` | `~/.edgellm/models` | Model storage directory |
| `EDGELLM_BIN` | `~/.local/bin` | Binary installation path |

## Troubleshooting

### "Command not found: edgellm"

Add to your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Build fails on macOS x86_64

Mojo doesn't support macOS x86_64 natively. Use Docker:
```bash
docker compose -f docker-compose.mojo.yml up --build
```

### Permission denied

```bash
chmod +x ~/.local/bin/edgellm
```

## For Package Maintainers

### Build Conda Package

```bash
pixi run build-package
```

### Publish to prefix.dev

```bash
export PREFIX_DEV_TOKEN=your-token
pixi run publish
```
