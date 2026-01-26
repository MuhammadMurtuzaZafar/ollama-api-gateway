#!/usr/bin/env python3
"""
Convert TMAC INT8 format to TM2 ternary format for Mojo inference.

TMAC Format (from quantize_bitnet.py):
- Magic: "TMAC"
- Header with config
- Per-tensor: name, dtype, shape, INT8 data, float scales

TM2 Format (expected by Mojo):
- Magic: "TM2\0"
- Header with config
- Per-row ternary weights (2-bit packed) + float16 scales
"""

import struct
import numpy as np
from pathlib import Path
import sys


def unpack_bitnet_base3(packed: np.ndarray, num_values: int) -> np.ndarray:
    """
    Unpack BitNet base-3 encoded weights to ternary {-1, 0, +1}.
    Each byte encodes 4 ternary values: w0 + 3*w1 + 9*w2 + 27*w3
    Values {0,1,2} map to {-1,0,+1}
    """
    packed_flat = packed.flatten().astype(np.int32)

    # Decode base-3
    w0 = packed_flat % 3
    w1 = (packed_flat // 3) % 3
    w2 = (packed_flat // 9) % 3
    w3 = (packed_flat // 27) % 3

    # Stack: each byte becomes 4 values
    unpacked = np.stack([w0, w1, w2, w3], axis=1).flatten()

    # Truncate to actual number of values
    unpacked = unpacked[:num_values]

    # Map {0,1,2} -> {-1,0,+1}
    ternary = unpacked.astype(np.int8) - 1

    return ternary


def read_tmac_format(path: str) -> tuple:
    """Read TMAC format file."""
    tensors = {}
    config = {}

    with open(path, "rb") as f:
        # Read magic
        magic = f.read(4)
        if magic != b"TMAC":
            raise ValueError(f"Invalid magic: {magic}")

        # Read header
        version = struct.unpack("I", f.read(4))[0]
        hidden_size = struct.unpack("I", f.read(4))[0]
        num_layers = struct.unpack("I", f.read(4))[0]
        num_heads = struct.unpack("I", f.read(4))[0]
        vocab_size = struct.unpack("I", f.read(4))[0]
        bits = struct.unpack("I", f.read(4))[0]
        group_size = struct.unpack("I", f.read(4))[0]

        config = {
            "version": version,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "vocab_size": vocab_size,
            "bits": bits,
            "group_size": group_size
        }

        print(f"Config: {config}")

        # Read tensors
        tensor_count = 0
        while True:
            try:
                # Name length
                name_len_bytes = f.read(4)
                if len(name_len_bytes) < 4:
                    break
                name_len = struct.unpack("I", name_len_bytes)[0]
                if name_len == 0 or name_len > 1000:
                    break

                # Name
                name = f.read(name_len).decode("utf-8")

                # Dtype length and value
                dtype_len = struct.unpack("I", f.read(4))[0]
                dtype = f.read(dtype_len).decode("utf-8")

                # Shape
                num_dims = struct.unpack("I", f.read(4))[0]
                shape = []
                for _ in range(num_dims):
                    shape.append(struct.unpack("I", f.read(4))[0])

                # Data
                data_size = struct.unpack("Q", f.read(8))[0]
                raw_data = f.read(data_size)

                # Scales
                scales_size = struct.unpack("Q", f.read(8))[0]
                scales = np.frombuffer(f.read(scales_size), dtype=np.float32)

                # Parse based on dtype
                if dtype == "bitnet":
                    # BitNet: base-3 packed ternary
                    packed = np.frombuffer(raw_data, dtype=np.uint8)
                    # Infer original shape from scales (scales.shape[0] = rows)
                    rows = len(scales)
                    cols = (len(packed) * 4) // rows  # 4 values per byte
                    num_values = rows * cols
                    ternary = unpack_bitnet_base3(packed, num_values)
                    data = ternary.reshape(rows, cols)
                    original_shape = (rows, cols)
                elif dtype == "int8":
                    data = np.frombuffer(raw_data, dtype=np.int8)
                    if len(shape) == 2:
                        data = data.reshape(shape)
                        original_shape = tuple(shape)
                    else:
                        original_shape = (len(scales), len(data) // len(scales)) if len(scales) > 0 else tuple(shape)
                        try:
                            data = data.reshape(original_shape)
                        except:
                            original_shape = tuple(shape)
                elif dtype == "fp16":
                    data = np.frombuffer(raw_data, dtype=np.float16)
                    original_shape = tuple(shape) if shape else (len(data),)
                else:
                    data = np.frombuffer(raw_data, dtype=np.int8)
                    original_shape = tuple(shape)

                tensors[name] = {
                    "dtype": dtype,
                    "shape": original_shape,
                    "data": data,
                    "scales": scales
                }

                tensor_count += 1
                print(f"  [{tensor_count}] {name}: shape={original_shape}, dtype={dtype}, scales={scales.shape}")

            except Exception as e:
                print(f"Error reading tensor at position {f.tell()}: {e}")
                import traceback
                traceback.print_exc()
                break

    return config, tensors


def process_tensor_for_tm2(tensor: dict) -> tuple:
    """
    Process a tensor for TM2 format output.

    For BitNet: data is already ternary {-1, 0, +1}, just need to repack
    For INT8: need to convert to ternary first
    """
    dtype = tensor["dtype"]
    data = tensor["data"]
    scales = tensor["scales"]

    if data.ndim == 1:
        # 1D data - try to infer shape from scales
        rows = len(scales) if len(scales) > 0 else 1
        cols = len(data) // rows
        data = data.reshape(rows, cols)

    rows, cols = data.shape

    if dtype == "bitnet":
        # Already ternary, just use directly with original scales
        ternary = data.astype(np.int8)
        if len(scales) == rows:
            row_scales = scales.astype(np.float32)
        else:
            # Broadcast scales
            row_scales = np.ones(rows, dtype=np.float32) * (scales[0] if len(scales) > 0 else 1.0)
        return ternary, row_scales

    elif dtype == "int8":
        # Convert INT8 to ternary
        row_scales = []
        ternary_rows = []

        for row in range(rows):
            row_data = data[row].astype(np.float32)

            # Get scale for this row
            if len(scales) == rows:
                row_scale = scales[row]
            elif len(scales) > rows:
                groups_per_row = len(scales) // rows
                row_scale = np.mean(scales[row * groups_per_row:(row + 1) * groups_per_row])
            else:
                row_scale = scales[0] if len(scales) > 0 else 1.0

            # Dequantize
            dequant = row_data * row_scale / 127.0

            # Threshold to ternary
            std = np.std(dequant)
            threshold = std * 0.5 if std > 0 else 0.3

            ternary = np.where(dequant > threshold, 1,
                              np.where(dequant < -threshold, -1, 0)).astype(np.int8)

            # Compute new scale
            non_zero_mask = ternary != 0
            if np.any(non_zero_mask):
                new_scale = np.mean(np.abs(dequant[non_zero_mask]))
            else:
                new_scale = row_scale

            ternary_rows.append(ternary)
            row_scales.append(new_scale)

        return np.array(ternary_rows, dtype=np.int8), np.array(row_scales, dtype=np.float32)

    else:
        # FP16 or other - shouldn't be called for weights, but handle gracefully
        # Quantize to ternary
        row_scales = []
        ternary_rows = []

        for row in range(rows):
            row_data = data[row].astype(np.float32)
            std = np.std(row_data)
            threshold = std * 0.5 if std > 0 else 0.1

            ternary = np.where(row_data > threshold, 1,
                              np.where(row_data < -threshold, -1, 0)).astype(np.int8)

            non_zero_mask = ternary != 0
            scale = np.mean(np.abs(row_data[non_zero_mask])) if np.any(non_zero_mask) else 1.0

            ternary_rows.append(ternary)
            row_scales.append(scale)

        return np.array(ternary_rows, dtype=np.int8), np.array(row_scales, dtype=np.float32)


def pack_ternary(ternary: np.ndarray) -> np.ndarray:
    """
    Pack ternary {-1, 0, +1} to 2-bit format.
    Encoding: 00=0, 01=+1, 11=-1
    4 values per byte
    """
    flat = ternary.flatten()

    # Pad to multiple of 4
    pad_len = (4 - len(flat) % 4) % 4
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len))

    # Reshape to groups of 4
    groups = flat.reshape(-1, 4)

    # Encode: -1->3, 0->0, +1->1
    encoded = np.where(groups == -1, 3, np.where(groups == 1, 1, 0)).astype(np.uint8)

    # Pack 4 values per byte (bits 0-1, 2-3, 4-5, 6-7)
    packed = (encoded[:, 0] |
              (encoded[:, 1] << 2) |
              (encoded[:, 2] << 4) |
              (encoded[:, 3] << 6))

    return packed.astype(np.uint8)


def write_tm2_format(path: str, config: dict, tensors: dict):
    """Write TM2 format for Mojo inference."""

    # Map tensor names to layer structure
    # Expected tensor names from quantize_bitnet.py:
    # embed.weight, layers.*.attn.q/k/v/o.weight, layers.*.ffn.gate/up/down.weight, etc.

    dim = config["hidden_size"]
    n_layers = config["num_layers"]
    n_heads = config["num_heads"]
    vocab_size = config["vocab_size"]

    # Infer other config values
    # SmolLM-135M: dim=576, hidden=1536, heads=9, kv_heads=3
    # These are typical ratios
    kv_heads = n_heads // 3 if n_heads >= 3 else n_heads
    hidden_dim = dim * 8 // 3  # Typical ratio

    # Try to infer from tensors
    for name, tensor in tensors.items():
        if "ffn" in name and "gate" in name:
            hidden_dim = tensor["shape"][0]
            break
        if "mlp" in name and "gate" in name:
            hidden_dim = tensor["shape"][0]
            break

    print(f"\nWriting TM2 format:")
    print(f"  dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(f"  n_heads={n_heads}, kv_heads={kv_heads}, vocab_size={vocab_size}")

    with open(path, "wb") as f:
        # Magic
        f.write(b"TM2\x00")

        # Config header (7 x int32)
        f.write(struct.pack("I", dim))
        f.write(struct.pack("I", hidden_dim))
        f.write(struct.pack("I", n_layers))
        f.write(struct.pack("I", n_heads))
        f.write(struct.pack("I", kv_heads))
        f.write(struct.pack("I", vocab_size))
        f.write(struct.pack("I", 2048))  # seq_len

        def write_ternary_weight(name_pattern: str, fallback_shape=None):
            """Find and write a ternary weight matrix."""
            # Try various naming patterns
            patterns = [name_pattern]
            if "." in name_pattern:
                # Also try alternate naming
                patterns.append(name_pattern.replace(".", "_"))

            tensor = None
            matched_name = None
            for pattern in patterns:
                for tname, tdata in tensors.items():
                    if pattern in tname or tname == pattern:
                        tensor = tdata
                        matched_name = tname
                        break
                if tensor:
                    break

            if tensor is None:
                print(f"    Warning: tensor '{name_pattern}' not found, using zeros")
                if fallback_shape:
                    rows, cols = fallback_shape
                else:
                    rows, cols = dim, dim
                # Write zero weights
                f.write(bytes([1]))  # quantized flag
                f.write(struct.pack("I", rows))
                f.write(struct.pack("I", cols))
                bytes_per_row = (cols + 3) // 4
                for _ in range(rows):
                    f.write(struct.pack("e", 0.0))  # float16 scale
                    f.write(bytes(bytes_per_row))  # zero data
                return

            # Process tensor to ternary format
            ternary, row_scales = process_tensor_for_tm2(tensor)
            rows, cols = ternary.shape

            # Write
            f.write(bytes([1]))  # quantized flag
            f.write(struct.pack("I", rows))
            f.write(struct.pack("I", cols))

            bytes_per_row = (cols + 3) // 4
            for row in range(rows):
                # Float16 scale
                f.write(struct.pack("e", float(row_scales[row])))
                # Packed ternary data
                packed = pack_ternary(ternary[row])
                f.write(packed.tobytes()[:bytes_per_row])

            print(f"    {name_pattern} -> {matched_name}: {rows}x{cols}, packed {rows * bytes_per_row} bytes")

        def write_float_vector(name_pattern: str, size: int):
            """Find and write a float vector."""
            tensor = None
            for tname, tdata in tensors.items():
                if name_pattern in tname:
                    tensor = tdata
                    break

            f.write(bytes([0]))  # not quantized flag

            if tensor is None:
                print(f"    Warning: vector '{name_pattern}' not found, using ones")
                for _ in range(size):
                    f.write(struct.pack("f", 1.0))
                return

            data = tensor["data"].flatten().astype(np.float32)
            for i in range(min(size, len(data))):
                f.write(struct.pack("f", float(data[i])))
            # Pad if needed
            for _ in range(max(0, size - len(data))):
                f.write(struct.pack("f", 1.0))

            print(f"    {name_pattern}: {size} floats")

        head_dim = dim // n_heads
        kv_dim = head_dim * kv_heads

        print("\nWriting embedding...")
        write_ternary_weight("embed", (vocab_size, dim))

        for layer in range(n_layers):
            print(f"\nLayer {layer}:")

            # Input norm
            write_float_vector(f"layers.{layer}.input_layernorm", dim)

            # Attention weights
            write_ternary_weight(f"layers.{layer}.self_attn.q_proj", (dim, dim))
            write_ternary_weight(f"layers.{layer}.self_attn.k_proj", (kv_dim, dim))
            write_ternary_weight(f"layers.{layer}.self_attn.v_proj", (kv_dim, dim))
            write_ternary_weight(f"layers.{layer}.self_attn.o_proj", (dim, dim))

            # Attention sub-norm
            write_float_vector(f"layers.{layer}.attn_norm", dim)

            # Post-attention norm
            write_float_vector(f"layers.{layer}.post_attention_layernorm", dim)

            # FFN weights
            write_ternary_weight(f"layers.{layer}.mlp.gate_proj", (hidden_dim, dim))
            write_ternary_weight(f"layers.{layer}.mlp.up_proj", (hidden_dim, dim))
            write_ternary_weight(f"layers.{layer}.mlp.down_proj", (dim, hidden_dim))

            # FFN sub-norm
            write_float_vector(f"layers.{layer}.ffn_norm", hidden_dim)

        print("\nWriting final norm and LM head...")
        write_float_vector("model.norm", dim)

        # For lm_head: first try lm_head tensor, then fall back to embed_tokens (tied embeddings)
        lm_head_tensor = None
        for tname, tdata in tensors.items():
            if "lm_head" in tname:
                lm_head_tensor = tdata
                break

        if lm_head_tensor is None:
            print("    lm_head not found, using tied embedding (model.embed_tokens)")
            # Use embed_tokens for tied embeddings
            for tname, tdata in tensors.items():
                if "embed_tokens" in tname:
                    lm_head_tensor = tdata
                    break

        if lm_head_tensor is not None:
            ternary, row_scales = process_tensor_for_tm2(lm_head_tensor)
            rows, cols = ternary.shape

            f.write(bytes([1]))  # quantized flag
            f.write(struct.pack("I", rows))
            f.write(struct.pack("I", cols))

            bytes_per_row = (cols + 3) // 4
            for row in range(rows):
                f.write(struct.pack("e", float(row_scales[row])))
                packed = pack_ternary(ternary[row])
                f.write(packed.tobytes()[:bytes_per_row])

            print(f"    lm_head: {rows}x{cols}, packed {rows * bytes_per_row} bytes")
        else:
            print("    Warning: Neither lm_head nor embed_tokens found, using zeros")
            write_ternary_weight("lm_head", (vocab_size, dim))

    print(f"\nWrote: {path}")
    print(f"Size: {Path(path).stat().st_size / 1024 / 1024:.1f} MB")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.tmac.bin> <output.tm2.bin>")
        print("\nConverts TMAC INT8 format to TM2 ternary format for Mojo inference.")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Reading TMAC format: {input_path}")
    config, tensors = read_tmac_format(input_path)

    print(f"\nFound {len(tensors)} tensors")

    write_tm2_format(output_path, config, tensors)


if __name__ == "__main__":
    main()
