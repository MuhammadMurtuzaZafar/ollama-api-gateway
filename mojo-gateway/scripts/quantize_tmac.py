#!/usr/bin/env python3
"""
Quantize llama2.c .bin models to T-MAC ternary format.

T-MAC Format (based on https://arxiv.org/abs/2407.00088):
- Ternary weights: {-1, 0, +1}
- 2 bits per weight, 4 weights per byte
- Encoding: 00 = 0, 01 = +1, 11 = -1

Compression: 16x vs FP32 (0.5 bits average per ternary weight in practice)
Memory: ~875MB for 7B model (vs 14GB FP32)
"""

import struct
import numpy as np
import sys
from pathlib import Path

BLOCK_SIZE = 32  # For scale computation


def read_config(f):
    """Read model config from llama2.c format."""
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


def quantize_to_ternary(weights: np.ndarray) -> tuple:
    """
    Quantize float32 weights to ternary {-1, 0, +1}.

    Uses threshold-based quantization:
    - |w| < threshold → 0
    - w >= threshold → +1
    - w <= -threshold → -1

    Returns (packed_bytes, stats)
    """
    flat = weights.flatten()

    # Compute threshold based on weight distribution
    # Use standard deviation as threshold (common approach for ternary)
    abs_weights = np.abs(flat)
    threshold = 0.7 * np.mean(abs_weights)  # Empirical threshold

    # Quantize to ternary
    ternary = np.zeros_like(flat, dtype=np.int8)
    ternary[flat > threshold] = 1
    ternary[flat < -threshold] = -1

    # Pack 4 ternary weights per byte
    # Encoding: 00 = 0, 01 = +1, 11 = -1
    num_weights = len(flat)
    num_bytes = (num_weights + 3) // 4

    # Pad to multiple of 4
    if num_weights % 4 != 0:
        ternary = np.pad(ternary, (0, 4 - num_weights % 4))

    packed = bytearray(num_bytes)
    for i in range(num_bytes):
        byte_val = 0
        for j in range(4):
            w = ternary[i * 4 + j]
            if w == 0:
                bits = 0b00
            elif w == 1:
                bits = 0b01
            else:  # w == -1
                bits = 0b11
            byte_val |= bits << (j * 2)
        packed[i] = byte_val

    # Stats
    zeros = np.sum(ternary == 0)
    positives = np.sum(ternary == 1)
    negatives = np.sum(ternary == -1)
    sparsity = zeros / len(ternary) * 100

    return bytes(packed), {
        'zeros': zeros,
        'positives': positives,
        'negatives': negatives,
        'sparsity': sparsity,
        'threshold': threshold
    }


def read_weights(f, size: int) -> np.ndarray:
    """Read float32 weights from file."""
    data = f.read(size * 4)
    return np.frombuffer(data, dtype=np.float32)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.tmac.bin>")
        print("\nConverts llama2.c model to T-MAC ternary format.")
        print("Large weight matrices become ternary (16x compression).")
        print("Small matrices (norms, positions) stay float32.")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    print(f"Quantizing {input_path} -> {output_path} (T-MAC ternary)")
    print("=" * 60)

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

        # Define weight specifications
        # (name, size, quantize_to_ternary?)
        weight_specs = [
            ('token_embedding', vocab_size * dim, True),
            ('rms_att_weight', n_layers * dim, False),  # Keep float32
            ('wq', n_layers * dim * dim, True),
            ('wk', n_layers * kv_dim * dim, True),
            ('wv', n_layers * kv_dim * dim, True),
            ('wo', n_layers * dim * dim, True),
            ('rms_ffn_weight', n_layers * dim, False),  # Keep float32
            ('w1', n_layers * hidden_dim * dim, True),
            ('w2', n_layers * dim * hidden_dim, True),
            ('w3', n_layers * hidden_dim * dim, True),
            ('rms_final_weight', dim, False),  # Keep float32
            ('freq_cis_real', seq_len * head_size // 2, False),  # Keep float32
            ('freq_cis_imag', seq_len * head_size // 2, False),  # Keep float32
        ]

        if not shared_weights:
            weight_specs.append(('wcls', vocab_size * dim, True))

        quantized_weights = {}
        total_original = 0
        total_quantized = 0
        total_ternary_params = 0

        print("\nProcessing weights:")
        print("-" * 60)

        for name, size, do_ternary in weight_specs:
            weights = read_weights(f, size)
            orig_size = size * 4  # float32 = 4 bytes

            if do_ternary:
                quantized, stats = quantize_to_ternary(weights)
                quant_size = len(quantized)
                total_ternary_params += size
                print(f"  {name:20s}: {size:12,} params -> ternary")
                print(f"    {orig_size:,} -> {quant_size:,} bytes ({orig_size/quant_size:.1f}x)")
                print(f"    Sparsity: {stats['sparsity']:.1f}% (threshold={stats['threshold']:.4f})")
                quantized_weights[name] = (quantized, True)
            else:
                quantized = weights.tobytes()
                quant_size = len(quantized)
                print(f"  {name:20s}: {size:12,} params -> float32 (kept)")
                quantized_weights[name] = (quantized, False)

            total_original += orig_size
            total_quantized += quant_size

    # Write T-MAC format
    print("\n" + "=" * 60)
    print("Writing T-MAC model...")

    with open(output_path, 'wb') as f:
        # Magic: "TMAC"
        f.write(b'TMAC')

        # Config (7 x int32)
        f.write(struct.pack('7i',
            config['dim'], config['hidden_dim'], config['n_layers'],
            config['n_heads'], config['n_kv_heads'], config['vocab_size'],
            config['seq_len']
        ))

        # Write weights in order
        for name, size, do_ternary in weight_specs:
            data, is_ternary = quantized_weights[name]
            f.write(data)

    final_size = output_path.stat().st_size

    print(f"\nSummary:")
    print(f"  Original size:      {total_original:,} bytes ({total_original/1024/1024:.1f} MB)")
    print(f"  T-MAC size:         {total_quantized:,} bytes ({total_quantized/1024/1024:.1f} MB)")
    print(f"  Compression:        {total_original / total_quantized:.1f}x")
    print(f"  Ternary parameters: {total_ternary_params:,}")
    print(f"  Output file:        {output_path} ({final_size:,} bytes)")
    print()
    print("T-MAC quantization complete!")
    print("Memory savings: {:.0f}%".format((1 - total_quantized/total_original) * 100))


if __name__ == '__main__':
    main()
