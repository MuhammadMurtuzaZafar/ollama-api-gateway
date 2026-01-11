# Mojo Development Environment for EdgeLLM
# For Intel Mac users (x86_64) since native Mojo only supports ARM64 on macOS

FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    clang \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | sh
ENV PATH="/root/.pixi/bin:$PATH"

# Set up working directory
WORKDIR /workspace

# Create pixi project with Mojo
RUN pixi init -c https://conda.modular.com/max-nightly/ -c conda-forge \
    && pixi add mojo

# Copy project files
COPY . /workspace/

# Build C kernel for x86_64 Linux
RUN if [ -f src/kernels/Makefile ]; then make -C src/kernels clean all; fi

# Default command
CMD ["pixi", "shell"]
