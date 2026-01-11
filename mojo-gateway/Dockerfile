# Mojo Gateway Dockerfile
# Multi-stage build for optimized production image

# ============================================
# Stage 1: Build Environment
# ============================================
FROM ubuntu:22.04 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Modular CLI and Mojo
RUN curl -ssL https://magic.modular.com | bash
ENV PATH="/root/.modular/bin:$PATH"

# Install MAX and Mojo packages
RUN magic install max mojo

# Set working directory
WORKDIR /app

# Copy project files
COPY mojoproject.toml .
COPY src/ src/

# Install dependencies
RUN magic install

# Build optimized binary
RUN magic run build

# ============================================
# Stage 2: Production Runtime
# ============================================
FROM ubuntu:22.04 AS runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 gateway
USER gateway

# Set working directory
WORKDIR /app

# Copy built binary from builder stage
COPY --from=builder /app/bin/gateway /app/gateway

# Environment variables
ENV GATEWAY_HOST=0.0.0.0
ENV GATEWAY_PORT=8080
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the gateway
CMD ["./gateway"]

# ============================================
# Stage 3: GPU Runtime (NVIDIA)
# ============================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime-gpu

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Modular runtime for MAX Engine
RUN curl -ssL https://magic.modular.com | bash
ENV PATH="/root/.modular/bin:$PATH"
RUN magic install max

# Create non-root user
RUN useradd -m -u 1000 gateway
USER gateway

WORKDIR /app

# Copy built binary
COPY --from=builder /app/bin/gateway /app/gateway

# GPU-specific environment
ENV GATEWAY_HOST=0.0.0.0
ENV GATEWAY_PORT=8080
ENV LOG_LEVEL=INFO
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["./gateway"]
