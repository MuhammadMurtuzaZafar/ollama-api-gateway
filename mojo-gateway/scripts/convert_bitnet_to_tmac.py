#!/usr/bin/env python3
"""
Convert Microsoft BitNet b1.58 model to T-MAC format for Mojo inference.

BitNet Format:
- Weights packed as uint8 using base-3 encoding
- 4 ternary values per byte: w0 + 3*w1 + 9*w2 + 27*w3
- Each value {0,1,2} maps to {-1,0,+1}
- Global scale per tensor

T-MAC Format:
- 2-bit encoding: 00=0, 01=+1, 11=-1
- 4 weights per byte
- Per-row scale (T-MAC v2)
"""

import struct
import numpy as np
from safetensors import safe_open
from pathlib import Path
import json
import sys


def unpack_bitnet_ternary(packed: np.ndarray) -> np.ndarray:
    """
    Unpack BitNet base-3 encoded weights to ternary {-1, 0, +1}.
    Each byte encodes 4 ternary values: w0 + 3*w1 + 9*w2 + 27*w3
    """
    packed_flat = packed.flatten().astype(np.int32)

    # Decode base-3
    w0 = packed_flat % 3
    w1 = (packed_flat // 3) % 3
    w2 = (packed_flat // 9) % 3
    w3 = (packed_flat // 27) % 3

    # Stack and reshape: each byte becomes 4 values
    unpacked = np.stack([w0, w1, w2, w3], axis=1).flatten()

    # Map {0,1,2} -> {-1,0,+1}
    ternary = unpacked.astype(np.int8) - 1

    return ternary


def pack_tmac_ternary(ternary: np.ndarray) -> np.ndarray:
    """
    Pack ternary weights into T-MAC 2-bit format.
    Encoding: 00=0, 01=+1, 11=-1
    4 weights per byte
    """
    # Ensure multiple of 4
    pad_len = (4 - len(ternary) % 4) % 4
    if pad_len > 0:
        ternary = np.pad(ternary, (0, pad_len))

    # Reshape to groups of 4
    ternary = ternary.reshape(-1, 4)

    # Encode: -1->3, 0->0, +1->1
    encoded = np.where(ternary == -1, 3, np.where(ternary == 1, 1, 0)).astype(np.uint8)

    # Pack 4 values per byte
    packed = (encoded[:, 0] |
              (encoded[:, 1] << 2) |
              (encoded[:, 2] << 4) |
              (encoded[:, 3] << 6))

    return packed.astype(np.uint8)


def convert_bitnet_layer_weights(safetensors_file, layer_idx: int) -> dict:
    """Extract and convert weights for one transformer layer."""
    import torch
    prefix = f"model.layers.{layer_idx}"

    with safe_open(safetensors_file, framework='pt') as f:
        weights = {}

        # Attention weights (packed ternary)
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            key = f"{prefix}.self_attn.{proj}.weight"
            scale_key = f"{prefix}.self_attn.{proj}.weight_scale"

            packed = f.get_tensor(key).numpy()
            scale = float(f.get_tensor(scale_key).to(torch.float32).numpy())

            # Unpack BitNet format
            ternary = unpack_bitnet_ternary(packed)

            # Get actual dimensions
            packed_rows, packed_cols = packed.shape
            actual_rows = packed_rows * 4
            ternary = ternary[:actual_rows * packed_cols].reshape(actual_rows, packed_cols)

            weights[proj] = {'ternary': ternary, 'scale': scale, 'shape': (actual_rows, packed_cols)}

        # FFN weights (packed ternary)
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            key = f"{prefix}.mlp.{proj}.weight"
            scale_key = f"{prefix}.mlp.{proj}.weight_scale"

            packed = f.get_tensor(key).numpy()
            scale = float(f.get_tensor(scale_key).to(torch.float32).numpy())

            ternary = unpack_bitnet_ternary(packed)
            packed_rows, packed_cols = packed.shape
            actual_rows = packed_rows * 4
            ternary = ternary[:actual_rows * packed_cols].reshape(actual_rows, packed_cols)

            weights[proj] = {'ternary': ternary, 'scale': scale, 'shape': (actual_rows, packed_cols)}

        # Norm weights (float)
        for norm in ['input_layernorm', 'post_attention_layernorm']:
            key = f"{prefix}.{norm}.weight"
            weights[norm] = f.get_tensor(key).to(torch.float32).numpy()

        # Sub-norms
        weights['attn_sub_norm'] = f.get_tensor(f"{prefix}.self_attn.attn_sub_norm.weight").to(torch.float32).numpy()
        weights['ffn_sub_norm'] = f.get_tensor(f"{prefix}.mlp.ffn_sub_norm.weight").to(torch.float32).numpy()

    return weights


def write_tmac_v2_weight(f, ternary: np.ndarray, scale: float, rows: int, cols: int):
    """Write a single weight matrix in T-MAC v2 format with per-row scales."""
    # Flag: quantized
    f.write(struct.pack('B', 1))
    # Dimensions
    f.write(struct.pack('ii', rows, cols))

    # For T-MAC v2, we use per-row scales
    # Since BitNet has global scale, we'll use that for all rows
    scale_f16 = np.float16(scale)

    for row in range(rows):
        # Write scale (float16)
        f.write(scale_f16.tobytes())
        # Write packed ternary row
        row_data = ternary[row]
        packed = pack_tmac_ternary(row_data)
        f.write(packed.tobytes())


def write_float32_weight(f, weight: np.ndarray):
    """Write float32 weight with flag."""
    f.write(struct.pack('B', 0))  # Not quantized flag
    f.write(weight.astype(np.float32).tobytes())


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output.tmac2.bin> [model_dir]")
        print("\nConverts Microsoft BitNet-2B to T-MAC v2 format.")
        sys.exit(1)

    output_path = Path(sys.argv[1])
    model_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./models/bitnet-2b")

    safetensors_file = model_dir / "model.safetensors"
    config_file = model_dir / "config.json"

    print(f"Converting BitNet model to T-MAC v2 format")
    print(f"Input: {model_dir}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Load config
    with open(config_file) as f:
        config = json.load(f)

    dim = config['hidden_size']  # 2560
    hidden_dim = config['intermediate_size']  # 6912
    n_layers = config['num_hidden_layers']  # 30
    n_heads = config['num_attention_heads']  # 20
    n_kv_heads = config['num_key_value_heads']  # 5
    vocab_size = config['vocab_size']  # 128256
    seq_len = config['max_position_embeddings']  # 4096

    head_size = dim // n_heads  # 128
    kv_dim = n_kv_heads * head_size  # 640

    print(f"Config:")
    print(f"  dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_size={head_size}")
    print(f"  vocab_size={vocab_size}, seq_len={seq_len}")
    print()

    # Open output file
    with open(output_path, 'wb') as out:
        # Write magic: "TM2\0"
        out.write(b'TM2\x00')

        # Write config (7 x int32)
        # Note: Using negative vocab_size to indicate no weight sharing
        out.write(struct.pack('7i', dim, hidden_dim, n_layers, n_heads, n_kv_heads, -vocab_size, seq_len))

        # Load embedding
        print("Processing embeddings...")
        with safe_open(safetensors_file, framework='pt') as f:
            import torch
            embed_tokens = f.get_tensor("model.embed_tokens.weight").to(torch.float32).numpy()

        # Embedding stays float32 (not ternary in BitNet)
        # But we'll quantize it to ternary for our format
        print(f"  embed_tokens: {embed_tokens.shape}")

        # Simple ternary quantization for embeddings
        threshold = 0.4 * np.mean(np.abs(embed_tokens))
        embed_ternary = np.zeros_like(embed_tokens, dtype=np.int8)
        embed_ternary[embed_tokens > threshold] = 1
        embed_ternary[embed_tokens < -threshold] = -1
        embed_scale = np.mean(np.abs(embed_tokens[embed_tokens != 0])) if np.any(embed_tokens != 0) else 1.0

        write_tmac_v2_weight(out, embed_ternary, embed_scale, vocab_size, dim)
        print(f"  Written embed_tokens: {vocab_size} x {dim}")

        # Process each layer
        for layer_idx in range(n_layers):
            print(f"Processing layer {layer_idx + 1}/{n_layers}...")

            weights = convert_bitnet_layer_weights(safetensors_file, layer_idx)

            # Write input_layernorm (float32)
            write_float32_weight(out, weights['input_layernorm'])

            # Write attention weights (ternary)
            # Q projection
            q = weights['q_proj']
            write_tmac_v2_weight(out, q['ternary'], q['scale'], q['shape'][0], q['shape'][1])

            # K projection
            k = weights['k_proj']
            write_tmac_v2_weight(out, k['ternary'], k['scale'], k['shape'][0], k['shape'][1])

            # V projection
            v = weights['v_proj']
            write_tmac_v2_weight(out, v['ternary'], v['scale'], v['shape'][0], v['shape'][1])

            # O projection
            o = weights['o_proj']
            write_tmac_v2_weight(out, o['ternary'], o['scale'], o['shape'][0], o['shape'][1])

            # Write attn_sub_norm (float32)
            write_float32_weight(out, weights['attn_sub_norm'])

            # Write post_attention_layernorm (float32)
            write_float32_weight(out, weights['post_attention_layernorm'])

            # Write FFN weights (ternary)
            # gate_proj (w1 equivalent)
            gate = weights['gate_proj']
            write_tmac_v2_weight(out, gate['ternary'], gate['scale'], gate['shape'][0], gate['shape'][1])

            # up_proj (w3 equivalent)
            up = weights['up_proj']
            write_tmac_v2_weight(out, up['ternary'], up['scale'], up['shape'][0], up['shape'][1])

            # down_proj (w2 equivalent)
            down = weights['down_proj']
            write_tmac_v2_weight(out, down['ternary'], down['scale'], down['shape'][0], down['shape'][1])

            # Write ffn_sub_norm (float32)
            write_float32_weight(out, weights['ffn_sub_norm'])

        # Final norm
        print("Processing final layers...")
        with safe_open(safetensors_file, framework='pt') as f:
            import torch
            final_norm = f.get_tensor("model.norm.weight").to(torch.float32).numpy()
        write_float32_weight(out, final_norm)

        # LM head (output projection) - same as embed_tokens for tied weights
        # But BitNet ties weights, so we need to handle this
        # We'll write the embedding again as the classifier
        write_tmac_v2_weight(out, embed_ternary, embed_scale, vocab_size, dim)
        print(f"  Written lm_head: {vocab_size} x {dim}")

    output_size = output_path.stat().st_size
    print()
    print("=" * 60)
    print(f"Conversion complete!")
    print(f"Output: {output_path}")
    print(f"Size: {output_size / 1024 / 1024:.1f} MB")
    print()
    print("Note: This model uses BitNet architecture which differs from LLaMA.")
    print("You'll need to use the BitNet-specific inference engine.")


if __name__ == '__main__':
    main()
