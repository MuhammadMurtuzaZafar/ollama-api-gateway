#!/usr/bin/env python3
"""
EdgeLLM T-MAC Converter

Converts a quantized model to the T-MAC binary format for EdgeLLM runtime.
This format is optimized for pshufb/tbl-based inference.

Usage:
    python convert_to_tmac.py \
        --input ./model.bitnet.bin \
        --output ./model.tmac2.bin

The T-MAC format stores:
    - Pre-computed lookup tables for each group
    - Packed ternary weights
    - RoPE frequencies
    - Layer normalization weights
"""

import argparse
import struct
from pathlib import Path
from typing import Dict, Any, Optional
import json

import numpy as np


def read_bitnet_file(path: str) -> Dict[str, Any]:
    """Read a BitNet quantized file."""
    tensors = {}

    with open(path, "rb") as f:
        # Read magic
        magic = f.read(4)
        if magic != b"TMAC":
            raise ValueError(f"Invalid file format: expected TMAC, got {magic}")

        # Read version
        version = struct.unpack("I", f.read(4))[0]

        # Read config
        hidden_size = struct.unpack("I", f.read(4))[0]
        num_layers = struct.unpack("I", f.read(4))[0]
        num_heads = struct.unpack("I", f.read(4))[0]
        vocab_size = struct.unpack("I", f.read(4))[0]

        # Read quantization config
        bits = struct.unpack("I", f.read(4))[0]
        group_size = struct.unpack("I", f.read(4))[0]

        config = {
            "version": version,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "vocab_size": vocab_size,
            "bits": bits,
            "group_size": group_size,
        }

        # Read tensors
        while True:
            try:
                # Read name
                name_len = struct.unpack("I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")

                # Read dtype
                dtype_len = struct.unpack("I", f.read(4))[0]
                dtype = f.read(dtype_len).decode("utf-8")

                # Read shape
                ndim = struct.unpack("I", f.read(4))[0]
                shape = tuple(struct.unpack("I", f.read(4))[0] for _ in range(ndim))

                # Read data
                data_size = struct.unpack("Q", f.read(8))[0]
                data = np.frombuffer(f.read(data_size), dtype=np.uint8)

                # Read scales
                scales_size = struct.unpack("Q", f.read(8))[0]
                scales = np.frombuffer(f.read(scales_size), dtype=np.float32)

                tensors[name] = {
                    "dtype": dtype,
                    "shape": shape,
                    "data": data,
                    "scales": scales,
                }
            except struct.error:
                break

    return {"config": config, "tensors": tensors}


def build_lut_for_group(activations: np.ndarray, group_size: int = 4) -> np.ndarray:
    """
    Build lookup table for a group of activations.

    For BitNet with group_size=4, we have 256 possible patterns.
    Each pattern is a 4-value combination of {-1, 0, +1}.
    """
    lut = np.zeros(256, dtype=np.float32)

    for pattern in range(256):
        sum_val = 0.0
        for j in range(group_size):
            val_enc = (pattern >> (j * 2)) & 0x3
            # Decode: 0 -> -1, 1 -> 0, 2 -> +1
            weight = -1.0 if val_enc == 0 else (0.0 if val_enc == 1 else 1.0)
            if j < len(activations):
                sum_val += weight * activations[j]
        lut[pattern] = sum_val

    return lut


def write_tmac2_header(f, config: Dict[str, Any]):
    """Write T-MAC v2 header."""
    # Magic: TMAC2
    f.write(b"TMC2")

    # Version
    f.write(struct.pack("I", 2))

    # Model config
    f.write(struct.pack("I", config["hidden_size"]))
    f.write(struct.pack("I", config["num_layers"]))
    f.write(struct.pack("I", config["num_heads"]))
    f.write(struct.pack("I", config["vocab_size"]))

    # Head dimension
    head_dim = config["hidden_size"] // config["num_heads"]
    f.write(struct.pack("I", head_dim))

    # Intermediate size (estimate: 4x hidden for FFN)
    intermediate_size = config["hidden_size"] * 4
    f.write(struct.pack("I", intermediate_size))

    # Quantization config
    f.write(struct.pack("I", config.get("bits", 2)))
    f.write(struct.pack("I", config.get("group_size", 4)))

    # RoPE config
    f.write(struct.pack("f", 10000.0))  # rope_theta
    f.write(struct.pack("I", 512))  # max_seq_len


def write_tensor_tmac2(f, name: str, data: np.ndarray, dtype: str):
    """Write tensor in T-MAC v2 format."""
    # Name
    name_bytes = name.encode("utf-8")
    f.write(struct.pack("H", len(name_bytes)))
    f.write(name_bytes)

    # Dtype code: 0=fp16, 1=fp32, 2=int8, 3=bitnet
    dtype_code = {"fp16": 0, "fp32": 1, "int8": 2, "bitnet": 3}.get(dtype, 0)
    f.write(struct.pack("B", dtype_code))

    # Shape
    shape = data.shape
    f.write(struct.pack("B", len(shape)))
    for dim in shape:
        f.write(struct.pack("I", dim))

    # Data
    f.write(struct.pack("Q", data.nbytes))
    f.write(data.tobytes())


def convert_to_tmac2(input_path: str, output_path: str):
    """Convert BitNet format to T-MAC v2."""
    print(f"Reading {input_path}...")
    model_data = read_bitnet_file(input_path)

    config = model_data["config"]
    tensors = model_data["tensors"]

    print(f"\nModel configuration:")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Heads: {config['num_heads']}")
    print(f"  Vocab: {config['vocab_size']}")

    print(f"\nWriting T-MAC v2 format to {output_path}...")

    with open(output_path, "wb") as f:
        # Write header
        write_tmac2_header(f, config)

        # Write number of tensors
        f.write(struct.pack("I", len(tensors)))

        # Write each tensor
        for name, tensor_data in tensors.items():
            # Combine data with scales for T-MAC format
            data = tensor_data["data"]
            dtype = tensor_data["dtype"]

            write_tensor_tmac2(f, name, data, dtype)

            # Write scales separately if present
            if len(tensor_data["scales"]) > 0:
                scales_name = f"{name}.scales"
                write_tensor_tmac2(f, scales_name, tensor_data["scales"], "fp32")

            print(f"  Written: {name} ({dtype})")

    # Calculate file size
    output_size = Path(output_path).stat().st_size

    print(f"\n" + "=" * 60)
    print(f"Conversion complete!")
    print(f"Output: {output_path}")
    print(f"Size: {output_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert to T-MAC format")

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input model file (BitNet format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output T-MAC file"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EdgeLLM T-MAC Converter")
    print("=" * 60)

    convert_to_tmac2(args.input, args.output)


if __name__ == "__main__":
    main()
