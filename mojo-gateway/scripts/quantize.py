#!/usr/bin/env python3
"""
Quantize llama2.c .bin models to Q8_0 format for efficient inference.

Q8_0 format (per block of 32 weights):
- 1 float16 scale (2 bytes)
- 32 int8 values (32 bytes)
- Total: 34 bytes per 32 weights (vs 128 bytes float32)
- Compression: 3.76x

Usage:
    python quantize.py stories110M.bin stories110M.q8.bin
"""

import struct
import numpy as np
import sys
from pathlib import Path

BLOCK_SIZE = 32  # Weights per quantization block


def read_config(f):
    """Read model config from .bin file."""
    config_data = f.read(7 * 4)  # 7 int32 values
    config = struct.unpack('7i', config_data)
    return {
        'dim': config[0],
        'hidden_dim': config[1],
        'n_layers': config[2],
        'n_heads': config[3],
        'n_kv_heads': config[4],
        'vocab_size': config[5],
        'seq_len': config[6],
    }


def quantize_block_q8(block: np.ndarray) -> bytes:
    """Quantize a block of 32 float32 values to Q8_0 format."""
    assert len(block) == BLOCK_SIZE

    # Find scale (max absolute value)
    max_abs = np.max(np.abs(block))
    if max_abs == 0:
        scale = np.float16(0.0)
        quant = np.zeros(BLOCK_SIZE, dtype=np.int8)
    else:
        scale = np.float16(max_abs / 127.0)
        # Quantize: round(value / scale) clamped to [-127, 127]
        quant = np.clip(np.round(block / float(scale)), -127, 127).astype(np.int8)

    # Pack: scale (2 bytes) + 32 int8 values (32 bytes)
    return scale.tobytes() + quant.tobytes()


def quantize_tensor(weights: np.ndarray) -> bytes:
    """Quantize entire tensor to Q8_0 format."""
    flat = weights.flatten()
    n_blocks = (len(flat) + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Pad to multiple of block size
    padded_len = n_blocks * BLOCK_SIZE
    if len(flat) < padded_len:
        flat = np.pad(flat, (0, padded_len - len(flat)))

    # Quantize each block
    result = bytearray()
    for i in range(n_blocks):
        block = flat[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
        result.extend(quantize_block_q8(block))

    return bytes(result)


def read_weights(f, size: int) -> np.ndarray:
    """Read float32 weights from file."""
    data = f.read(size * 4)
    return np.frombuffer(data, dtype=np.float32)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.q8.bin>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    print(f"Quantizing {input_path} -> {output_path}")

    with open(input_path, 'rb') as f:
        # Read config
        config = read_config(f)
        print(f"Config: {config}")

        dim = config['dim']
        hidden_dim = config['hidden_dim']
        n_layers = config['n_layers']
        n_heads = config['n_heads']
        n_kv_heads = config['n_kv_heads']
        vocab_size = abs(config['vocab_size'])
        seq_len = config['seq_len']
        head_size = dim // n_heads
        kv_dim = (n_kv_heads * dim) // n_heads

        shared_weights = config['vocab_size'] > 0

        # Define weight sizes
        weight_specs = [
            ('token_embedding', vocab_size * dim),
            ('rms_att_weight', n_layers * dim),
            ('wq', n_layers * dim * dim),
            ('wk', n_layers * kv_dim * dim),
            ('wv', n_layers * kv_dim * dim),
            ('wo', n_layers * dim * dim),
            ('rms_ffn_weight', n_layers * dim),
            ('w1', n_layers * hidden_dim * dim),
            ('w2', n_layers * dim * hidden_dim),
            ('w3', n_layers * hidden_dim * dim),
            ('rms_final_weight', dim),
            ('freq_cis_real', seq_len * head_size // 2),
            ('freq_cis_imag', seq_len * head_size // 2),
        ]

        if not shared_weights:
            weight_specs.append(('wcls', vocab_size * dim))

        # Read and quantize weights
        quantized_weights = {}
        total_original = 0
        total_quantized = 0

        for name, size in weight_specs:
            print(f"  Quantizing {name}: {size:,} params...", end=' ')
            weights = read_weights(f, size)

            # Don't quantize small tensors (norm weights, freq_cis)
            if 'rms_' in name or 'freq_' in name:
                # Keep as float32
                quantized = weights.tobytes()
                is_quantized = False
            else:
                quantized = quantize_tensor(weights)
                is_quantized = True

            quantized_weights[name] = (quantized, is_quantized, size)

            orig_size = size * 4
            quant_size = len(quantized)
            total_original += orig_size
            total_quantized += quant_size

            ratio = orig_size / quant_size if quant_size > 0 else 0
            print(f"{orig_size:,} -> {quant_size:,} bytes ({ratio:.2f}x)")

    # Write quantized file
    with open(output_path, 'wb') as f:
        # Write magic number and version
        f.write(b'Q8V1')  # Magic: Q8 version 1

        # Write config (same format)
        f.write(struct.pack('7i',
            config['dim'], config['hidden_dim'], config['n_layers'],
            config['n_heads'], config['n_kv_heads'], config['vocab_size'],
            config['seq_len']
        ))

        # Write quantization info
        f.write(struct.pack('i', BLOCK_SIZE))

        # Write weights
        for name, size in weight_specs:
            data, is_quantized, _ = quantized_weights[name]
            # Write flag: 1 = quantized, 0 = float32
            f.write(struct.pack('B', 1 if is_quantized else 0))
            f.write(data)

    print(f"\nTotal: {total_original:,} -> {total_quantized:,} bytes")
    print(f"Compression: {total_original / total_quantized:.2f}x")
    print(f"Output: {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == '__main__':
    main()
