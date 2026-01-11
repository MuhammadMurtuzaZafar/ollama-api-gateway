#!/usr/bin/env python3
"""
EdgeLLM BitNet Quantization Script

Quantizes a model to BitNet 1.58-bit format.
BitNet uses ternary weights {-1, 0, +1} for 10x compression.

Usage:
    python quantize_bitnet.py \
        --input ./merged_model \
        --output ./model.bitnet.bin

Requirements:
    pip install torch transformers numpy
"""

import argparse
import struct
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def quantize_weight_bitnet(weight: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Quantize a weight tensor to BitNet 1.58-bit.

    BitNet uses ternary values: {-1, 0, +1}
    Encoded as 2 bits per value: 00=-1, 01=0, 10=+1

    Args:
        weight: Input weight tensor (2D)

    Returns:
        packed: Packed ternary weights (uint8)
        scales: Per-row scaling factors
    """
    # Ensure 2D
    original_shape = weight.shape
    if weight.dim() == 1:
        weight = weight.unsqueeze(0)

    # Compute per-row scale (absmax)
    scales = weight.abs().max(dim=-1, keepdim=True)[0]
    scales = torch.clamp(scales, min=1e-8)

    # Normalize to [-1, 1]
    normalized = weight / scales

    # Quantize to {-1, 0, +1} using thresholds
    # Values > 0.5 -> 1, values < -0.5 -> -1, else -> 0
    quantized = torch.zeros_like(normalized, dtype=torch.int8)
    quantized[normalized > 0.5] = 1
    quantized[normalized < -0.5] = -1

    # Pack 4 ternary values into 1 byte
    # Each value uses 2 bits: -1=00, 0=01, +1=10
    flat = quantized.flatten()
    packed_len = (len(flat) + 3) // 4
    packed = np.zeros(packed_len, dtype=np.uint8)

    for i in range(len(flat)):
        val = flat[i].item()
        # Encoding: -1 -> 0, 0 -> 1, +1 -> 2
        encoded = 1 if val == 0 else (0 if val == -1 else 2)
        byte_idx = i // 4
        bit_offset = (i % 4) * 2
        packed[byte_idx] |= (encoded << bit_offset)

    return packed, scales.squeeze(-1).view(-1)


def quantize_embedding(weight: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Quantize embedding layer to INT8.
    Embeddings are kept at higher precision for better quality.
    """
    # Per-row scale
    scales = weight.abs().max(dim=-1, keepdim=True)[0]
    scales = torch.clamp(scales, min=1e-8)

    # Quantize to INT8
    quantized = (weight / scales * 127).round().clamp(-128, 127).to(torch.int8)

    return quantized.numpy(), scales.squeeze(-1)


def write_tmac_header(f, config, version=2):
    """Write T-MAC file header."""
    # Magic number
    f.write(b"TMAC")

    # Version
    f.write(struct.pack("I", version))

    # Model configuration
    f.write(struct.pack("I", config.hidden_size))
    f.write(struct.pack("I", config.num_hidden_layers))
    f.write(struct.pack("I", config.num_attention_heads))
    f.write(struct.pack("I", config.vocab_size))

    # Quantization config
    f.write(struct.pack("I", 2))  # bits (1.58 -> stored as 2)
    f.write(struct.pack("I", 4))  # group_size


def write_tensor(f, name: str, data: np.ndarray, scales: torch.Tensor, dtype: str):
    """Write a quantized tensor to file."""
    # Name
    name_bytes = name.encode("utf-8")
    f.write(struct.pack("I", len(name_bytes)))
    f.write(name_bytes)

    # Data type
    dtype_bytes = dtype.encode("utf-8")
    f.write(struct.pack("I", len(dtype_bytes)))
    f.write(dtype_bytes)

    # Shape
    shape = data.shape
    f.write(struct.pack("I", len(shape)))
    for dim in shape:
        f.write(struct.pack("I", dim))

    # Data
    f.write(struct.pack("Q", data.nbytes))
    f.write(data.tobytes())

    # Scales
    scales_np = scales.float().numpy()
    f.write(struct.pack("Q", scales_np.nbytes))
    f.write(scales_np.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Quantize model to BitNet 1.58-bit")

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input model path (HuggingFace format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path (.tmac2.bin)"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Calibration data (JSONL) for better quantization"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of calibration samples"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EdgeLLM BitNet Quantization")
    print("=" * 60)

    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")

    # Load model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(args.input)
    model = AutoModelForCausalLM.from_pretrained(
        args.input,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.input)

    print(f"\nModel configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Heads: {config.num_attention_heads}")
    print(f"  Vocab size: {config.vocab_size}")

    # Calculate original size
    total_params = sum(p.numel() for p in model.parameters())
    original_size = total_params * 2  # FP16

    print(f"\nOriginal size (FP16): {original_size / 1024 / 1024:.1f} MB")
    print(f"Total parameters: {total_params:,}")

    # Quantize
    print("\nQuantizing weights...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantized_size = 0

    with open(args.output, "wb") as f:
        # Write header
        write_tmac_header(f, config)

        # Quantize each layer
        for name, param in model.named_parameters():
            if param.numel() == 0:
                continue

            if "embed" in name.lower() or "lm_head" in name.lower():
                # Embeddings: INT8
                packed, scales = quantize_embedding(param.data)
                dtype = "int8"
                print(f"  {name}: INT8 ({param.shape})")
            elif "weight" in name and param.dim() >= 2:
                # Linear layers: BitNet
                packed, scales = quantize_weight_bitnet(param.data)
                dtype = "bitnet"
                print(f"  {name}: BitNet ({param.shape})")
            else:
                # Biases, norms: FP16
                packed = param.data.half().numpy()
                scales = torch.ones(1)
                dtype = "fp16"
                print(f"  {name}: FP16 ({param.shape})")

            write_tensor(f, name, packed, scales, dtype)
            quantized_size += packed.nbytes

    # Final stats
    final_size = output_path.stat().st_size
    compression = original_size / final_size

    print("\n" + "=" * 60)
    print("Quantization Complete!")
    print("=" * 60)
    print(f"\nOriginal size (FP16): {original_size / 1024 / 1024:.1f} MB")
    print(f"Quantized size: {final_size / 1024 / 1024:.1f} MB")
    print(f"Compression ratio: {compression:.1f}x")
    print(f"\nOutput: {args.output}")
    print(f"\nNext steps:")
    print(f"  Test: edgellm benchmark --model {args.output}")
    print(f"  Deploy: edgellm serve --model {args.output}")


if __name__ == "__main__":
    main()
