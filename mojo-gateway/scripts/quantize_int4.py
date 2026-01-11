#!/usr/bin/env python3
"""
Quantize llama2.c .bin models to Q4_0 format (int4).

Q4_0 format (per block of 32 weights):
- 1 float16 scale (2 bytes)
- 16 bytes of packed int4 values (32 x 4-bit = 16 bytes)
- Total: 18 bytes per 32 weights (vs 128 bytes float32)
- Compression: 7.1x
"""

import struct
import numpy as np
import sys
from pathlib import Path

BLOCK_SIZE = 32


def read_config(f):
    config_data = f.read(7 * 4)
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


def quantize_block_q4(block: np.ndarray) -> bytes:
    """Quantize 32 float32 values to Q4_0 format."""
    assert len(block) == BLOCK_SIZE

    max_abs = np.max(np.abs(block))
    if max_abs == 0:
        scale = np.float16(0.0)
        packed = bytes(16)
    else:
        # Scale to fit in [-8, 7] range (signed 4-bit)
        scale = np.float16(max_abs / 7.0)
        # Quantize to [-8, 7] then shift to [0, 15] for packing
        quant = np.clip(np.round(block / float(scale)), -8, 7).astype(np.int8)
        # Shift to unsigned [0, 15] for packing
        quant_u = (quant + 8).astype(np.uint8)

        # Pack two 4-bit values per byte
        packed = bytearray(16)
        for i in range(16):
            low = quant_u[i * 2] & 0x0F
            high = quant_u[i * 2 + 1] & 0x0F
            packed[i] = low | (high << 4)
        packed = bytes(packed)

    return scale.tobytes() + packed


def quantize_tensor_q4(weights: np.ndarray) -> bytes:
    """Quantize entire tensor to Q4_0 format."""
    flat = weights.flatten()
    n_blocks = (len(flat) + BLOCK_SIZE - 1) // BLOCK_SIZE

    padded_len = n_blocks * BLOCK_SIZE
    if len(flat) < padded_len:
        flat = np.pad(flat, (0, padded_len - len(flat)))

    result = bytearray()
    for i in range(n_blocks):
        block = flat[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
        result.extend(quantize_block_q4(block))

    return bytes(result)


def read_weights(f, size: int) -> np.ndarray:
    data = f.read(size * 4)
    return np.frombuffer(data, dtype=np.float32)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.q4.bin>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    print(f"Quantizing {input_path} -> {output_path} (int4)")

    with open(input_path, 'rb') as f:
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

        weight_specs = [
            ('token_embedding', vocab_size * dim, True),
            ('rms_att_weight', n_layers * dim, False),
            ('wq', n_layers * dim * dim, True),
            ('wk', n_layers * kv_dim * dim, True),
            ('wv', n_layers * kv_dim * dim, True),
            ('wo', n_layers * dim * dim, True),
            ('rms_ffn_weight', n_layers * dim, False),
            ('w1', n_layers * hidden_dim * dim, True),
            ('w2', n_layers * dim * hidden_dim, True),
            ('w3', n_layers * hidden_dim * dim, True),
            ('rms_final_weight', dim, False),
            ('freq_cis_real', seq_len * head_size // 2, False),
            ('freq_cis_imag', seq_len * head_size // 2, False),
        ]

        if not shared_weights:
            weight_specs.append(('wcls', vocab_size * dim, True))

        quantized_weights = {}
        total_original = 0
        total_quantized = 0

        for name, size, do_quantize in weight_specs:
            print(f"  Processing {name}: {size:,} params...", end=' ')
            weights = read_weights(f, size)

            if do_quantize:
                quantized = quantize_tensor_q4(weights)
            else:
                quantized = weights.tobytes()

            quantized_weights[name] = (quantized, do_quantize, size)

            orig_size = size * 4
            quant_size = len(quantized)
            total_original += orig_size
            total_quantized += quant_size

            ratio = orig_size / quant_size if quant_size > 0 else 0
            print(f"{orig_size:,} -> {quant_size:,} bytes ({ratio:.2f}x)")

    with open(output_path, 'wb') as f:
        f.write(b'Q4V1')  # Magic
        f.write(struct.pack('7i',
            config['dim'], config['hidden_dim'], config['n_layers'],
            config['n_heads'], config['n_kv_heads'], config['vocab_size'],
            config['seq_len']
        ))
        f.write(struct.pack('i', BLOCK_SIZE))

        for name, size, do_quantize in weight_specs:
            data, is_quantized, _ = quantized_weights[name]
            f.write(struct.pack('B', 1 if is_quantized else 0))
            f.write(data)

    print(f"\nTotal: {total_original:,} -> {total_quantized:,} bytes")
    print(f"Compression: {total_original / total_quantized:.2f}x")
    print(f"Output: {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == '__main__':
    main()
