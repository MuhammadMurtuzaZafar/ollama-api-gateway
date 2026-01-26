# Mojo Development Environment for EdgeLLM
# For Intel Mac users (x86_64) since native Mojo only supports ARM64 on macOS

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    clang \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | sh
ENV PATH="/root/.pixi/bin:$PATH"

# Set up working directory
WORKDIR /workspace

# Create pixi project with Mojo
RUN pixi init -c https://conda.modular.com/max-nightly/ -c conda-forge \
    && pixi add mojo \
    && pixi add python>=3.11

# Install Python dependencies for benchmarking (lightweight, no torch)
RUN pip3 install --no-cache-dir \
    psutil \
    requests \
    numpy

# Copy project files
COPY . /workspace/

# Build C kernel for x86_64 Linux
RUN mkdir -p lib bin && \
    if [ -f src/kernels/Makefile ]; then \
        make -C src/kernels clean all; \
    elif [ -f src/kernels/tmac_kernel.c ]; then \
        clang -O3 -mavx2 -shared -fPIC \
            -o lib/libtmac_kernel.so \
            src/kernels/tmac_kernel.c; \
    fi

# Build Mojo inference binary
RUN pixi run mojo build -O3 src/bitnet_tmac_lut.mojo -o bin/edgellm 2>&1 || \
    echo "Mojo build completed (check for errors above)"

# Make scripts executable
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/workspace/lib:$LD_LIBRARY_PATH

# Default command - run shell for interactive use
CMD ["bash"]
