#!/usr/bin/env python3
"""
Quantize llama2.c .bin models to T-MAC v2 format with per-row scales.

T-MAC v2 Improvements:
- Per-row scale factors (float16) to preserve magnitude
- Optimized threshold based on weight distribution
- Output = scale × ternary_value (recovers magnitude)

Format per matrix row:
- 1 × float16 scale (2 bytes)
- N/4 bytes of packed ternary weights

This achieves ~12x compression while maintaining quality.
"""

import struct
import numpy as np
import sys
from pathlib import Path


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


def quantize_row_ternary_scaled(row: np.ndarray) -> tuple:
    """
    Quantize a single row to scaled ternary.

    Returns: (scale_bytes, packed_ternary_bytes)

    The key insight: we store scale = mean(|non-zero weights|)
    Then ternary values {-1, 0, +1} × scale ≈ original values
    """
    # Compute scale as mean of absolute values (excluding near-zero)
    abs_row = np.abs(row)
    threshold = 0.4 * np.mean(abs_row)  # Lower threshold = fewer zeros = better quality

    # Non-zero weights for scale computation
    significant = abs_row[abs_row > threshold]
    if len(significant) > 0:
        scale = np.float16(np.mean(significant))
    else:
        scale = np.float16(np.mean(abs_row) + 1e-6)

    # Quantize to ternary using threshold
    ternary = np.zeros_like(row, dtype=np.int8)
    ternary[row > threshold] = 1
    ternary[row < -threshold] = -1

    # Pack 4 ternary weights per byte
    num_weights = len(row)
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

    return scale.tobytes(), bytes(packed)


def quantize_matrix_ternary_scaled(weights: np.ndarray, rows: int, cols: int) -> tuple:
    """
    Quantize a matrix with per-row scales.

    Returns: (all_bytes, stats)

    Format: [scale0][ternary0][scale1][ternary1]...
    """
    flat = weights.flatten()
    matrix = flat.reshape(rows, cols)

    result = bytearray()
    total_zeros = 0
    total_ones = 0
    total_neg_ones = 0

    for row_idx in range(rows):
        row = matrix[row_idx]
        scale_bytes, ternary_bytes = quantize_row_ternary_scaled(row)
        result.extend(scale_bytes)
        result.extend(ternary_bytes)

        # Stats
        abs_row = np.abs(row)
        threshold = 0.4 * np.mean(abs_row)
        total_zeros += np.sum(np.abs(row) <= threshold)
        total_ones += np.sum(row > threshold)
        total_neg_ones += np.sum(row < -threshold)

    total = rows * cols
    stats = {
        'zeros': total_zeros,
        'positives': total_ones,
        'negatives': total_neg_ones,
        'sparsity': total_zeros / total * 100
    }

    return bytes(result), stats


def read_weights(f, size: int) -> np.ndarray:
    """Read float32 weights from file."""
    data = f.read(size * 4)
    return np.frombuffer(data, dtype=np.float32)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.tmac2.bin>")
        print("\nT-MAC v2: Ternary with per-row scales for better quality.")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    print(f"Quantizing {input_path} -> {output_path} (T-MAC v2 with scales)")
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

        # Weight specifications: (name, size, rows, cols, quantize?)
        # For matmul: output = weight @ input, so weight is (output_dim, input_dim)
        weight_specs = [
            ('token_embedding', vocab_size * dim, vocab_size, dim, True),
            ('rms_att_weight', n_layers * dim, n_layers, dim, False),
            ('wq', n_layers * dim * dim, n_layers * dim, dim, True),  # (n_layers*dim, dim)
            ('wk', n_layers * kv_dim * dim, n_layers * kv_dim, dim, True),  # (n_layers*kv_dim, dim)
            ('wv', n_layers * kv_dim * dim, n_layers * kv_dim, dim, True),
            ('wo', n_layers * dim * dim, n_layers * dim, dim, True),  # (n_layers*dim, dim)
            ('rms_ffn_weight', n_layers * dim, n_layers, dim, False),
            ('w1', n_layers * hidden_dim * dim, n_layers * hidden_dim, dim, True),
            ('w2', n_layers * dim * hidden_dim, n_layers * dim, hidden_dim, True),
            ('w3', n_layers * hidden_dim * dim, n_layers * hidden_dim, dim, True),
            ('rms_final_weight', dim, 1, dim, False),
            ('freq_cis_real', seq_len * head_size // 2, seq_len, head_size // 2, False),
            ('freq_cis_imag', seq_len * head_size // 2, seq_len, head_size // 2, False),
        ]

        if not shared_weights:
            weight_specs.append(('wcls', vocab_size * dim, vocab_size, dim, True))

        quantized_weights = {}
        total_original = 0
        total_quantized = 0

        print("\nProcessing weights:")
        print("-" * 60)

        for name, size, rows, cols, do_quantize in weight_specs:
            weights = read_weights(f, size)
            orig_size = size * 4

            if do_quantize:
                quantized, stats = quantize_matrix_ternary_scaled(weights, rows, cols)
                quant_size = len(quantized)

                # Calculate compression
                # Per row: 2 bytes scale + cols/4 bytes ternary
                expected_size = rows * (2 + (cols + 3) // 4)

                print(f"  {name:20s}: {rows:6d} rows × {cols:4d} cols")
                print(f"    {orig_size:,} -> {quant_size:,} bytes ({orig_size/quant_size:.1f}x)")
                print(f"    Sparsity: {stats['sparsity']:.1f}%")
                quantized_weights[name] = (quantized, True, rows, cols)
            else:
                quantized = weights.tobytes()
                quant_size = len(quantized)
                print(f"  {name:20s}: {size:,} params -> float32 (kept)")
                quantized_weights[name] = (quantized, False, rows, cols)

            total_original += orig_size
            total_quantized += quant_size

    # Write T-MAC v2 format
    print("\n" + "=" * 60)
    print("Writing T-MAC v2 model...")

    with open(output_path, 'wb') as f:
        # Magic: "TM2\0" (T-MAC version 2)
        f.write(b'TM2\x00')

        # Config (7 x int32)
        f.write(struct.pack('7i',
            config['dim'], config['hidden_dim'], config['n_layers'],
            config['n_heads'], config['n_kv_heads'], config['vocab_size'],
            config['seq_len']
        ))

        # Write weights in order with metadata
        for name, size, rows, cols, do_quantize in weight_specs:
            data, is_quantized, r, c = quantized_weights[name]
            # Write flag + rows + cols for quantized matrices
            if is_quantized:
                f.write(struct.pack('B', 1))  # quantized flag
                f.write(struct.pack('ii', r, c))  # dimensions
            else:
                f.write(struct.pack('B', 0))  # not quantized
            f.write(data)

    final_size = output_path.stat().st_size

    print(f"\nSummary:")
    print(f"  Original size:      {total_original:,} bytes ({total_original/1024/1024:.1f} MB)")
    print(f"  T-MAC v2 size:      {total_quantized:,} bytes ({total_quantized/1024/1024:.1f} MB)")
    print(f"  Compression:        {total_original / total_quantized:.1f}x")
    print(f"  Output file:        {output_path} ({final_size:,} bytes)")
    print()
    print("T-MAC v2 quantization complete!")
    print(f"Memory savings: {(1 - total_quantized/total_original) * 100:.0f}%")


if __name__ == '__main__':
    main()
