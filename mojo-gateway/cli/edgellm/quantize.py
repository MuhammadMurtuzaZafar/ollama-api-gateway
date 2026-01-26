"""
EdgeLLM Quantization Module.

Converts models to BitNet 1.58-bit and T-MAC format.
"""

import click
from pathlib import Path
from typing import Optional
import struct


def run_quantize(
    input_path: str,
    output_path: str,
    format: str = "bitnet",
    calibration_data: Optional[str] = None,
    num_samples: int = 128,
):
    """Quantize model to specified format."""
    click.echo("\n" + "=" * 60)
    click.echo("EdgeLLM Quantization")
    click.echo("=" * 60)

    click.echo(f"\nInput: {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Format: {format}")

    if format == "bitnet":
        quantize_to_bitnet(input_path, output_path, calibration_data, num_samples)
    elif format == "int4":
        quantize_to_int4(input_path, output_path, calibration_data, num_samples)
    elif format == "int8":
        quantize_to_int8(input_path, output_path)
    elif format == "fp16":
        convert_to_fp16(input_path, output_path)
    else:
        raise click.ClickException(f"Unknown format: {format}")

    click.echo(f"\nQuantization complete! Output: {output_path}")


def quantize_to_bitnet(
    input_path: str,
    output_path: str,
    calibration_data: Optional[str],
    num_samples: int,
):
    """Quantize to BitNet 1.58-bit format."""
    click.echo("\nQuantizing to BitNet 1.58-bit...")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise click.ClickException(
            "Missing dependencies. Run: pip install torch transformers"
        )

    # Load model
    click.echo(f"Loading model from {input_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        input_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(input_path)

    # Get model config
    config = model.config
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    vocab_size = config.vocab_size

    click.echo(f"\nModel config:")
    click.echo(f"  Hidden size: {hidden_size}")
    click.echo(f"  Layers: {num_layers}")
    click.echo(f"  Heads: {num_heads}")
    click.echo(f"  Vocab size: {vocab_size}")

    # Prepare output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Write T-MAC format
    click.echo(f"\nWriting T-MAC format to {output_path}...")

    with open(output_path, "wb") as f:
        # Magic number: "TMAC"
        f.write(b"TMAC")

        # Version
        f.write(struct.pack("I", 2))  # Version 2

        # Model config
        f.write(struct.pack("I", hidden_size))
        f.write(struct.pack("I", num_layers))
        f.write(struct.pack("I", num_heads))
        f.write(struct.pack("I", vocab_size))

        # Quantization config
        f.write(struct.pack("I", 2))  # bits = 1.58 (stored as 2)
        f.write(struct.pack("I", 4))  # group_size = 4

        # Quantize and write weights
        total_params = 0
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                quantized, scales = quantize_weight_bitnet(param.data)
                total_params += param.numel()

                # Write layer info
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("I", len(name_bytes)))
                f.write(name_bytes)

                # Write shape
                f.write(struct.pack("I", len(param.shape)))
                for dim in param.shape:
                    f.write(struct.pack("I", dim))

                # Write quantized weights
                f.write(quantized.tobytes())

                # Write scales
                f.write(scales.numpy().tobytes())

        click.echo(f"\nTotal parameters: {total_params:,}")

    # Calculate compression
    original_size = total_params * 2  # FP16 = 2 bytes
    compressed_size = output.stat().st_size
    compression_ratio = original_size / compressed_size

    click.echo(f"Original size (FP16): {original_size / 1e6:.1f} MB")
    click.echo(f"Compressed size: {compressed_size / 1e6:.1f} MB")
    click.echo(f"Compression ratio: {compression_ratio:.1f}x")


def quantize_weight_bitnet(weight):
    """
    Quantize a weight tensor to BitNet 1.58-bit.

    BitNet uses ternary values: {-1, 0, +1}
    Encoded as 2 bits per value: 00=-1, 01=0, 10=+1
    """
    import torch
    import numpy as np

    # Compute per-row scale (absmax)
    scales = weight.abs().max(dim=-1, keepdim=True)[0]
    scales = torch.clamp(scales, min=1e-8)

    # Normalize to [-1, 1]
    normalized = weight / scales

    # Quantize to {-1, 0, +1}
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
        encoded = 1 if val == 0 else (0 if val == -1 else 2)
        byte_idx = i // 4
        bit_offset = (i % 4) * 2
        packed[byte_idx] |= (encoded << bit_offset)

    return packed, scales.squeeze(-1)


def quantize_to_int4(
    input_path: str,
    output_path: str,
    calibration_data: Optional[str],
    num_samples: int,
):
    """Quantize to INT4 format (GPTQ-style)."""
    click.echo("\nQuantizing to INT4...")

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise click.ClickException(
            "Missing dependencies. Run: pip install torch transformers"
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        input_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    click.echo("INT4 quantization: Writing weights...")

    # Simple min-max quantization to INT4
    with open(output_path, "wb") as f:
        f.write(b"TMAC")
        f.write(struct.pack("I", 1))  # Version 1 (INT4)

        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                # Quantize to INT4
                min_val = param.min()
                max_val = param.max()
                scale = (max_val - min_val) / 15

                quantized = ((param - min_val) / scale).round().clamp(0, 15).to(torch.uint8)

                # Pack 2 INT4 values per byte
                flat = quantized.flatten()
                packed = []
                for i in range(0, len(flat), 2):
                    low = flat[i].item()
                    high = flat[i + 1].item() if i + 1 < len(flat) else 0
                    packed.append(low | (high << 4))

                f.write(bytes(packed))

    click.echo("INT4 quantization complete!")


def quantize_to_int8(input_path: str, output_path: str):
    """Quantize to INT8 format."""
    click.echo("\nQuantizing to INT8...")

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise click.ClickException(
            "Missing dependencies. Run: pip install torch transformers"
        )

    model = AutoModelForCausalLM.from_pretrained(
        input_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    with open(output_path, "wb") as f:
        f.write(b"TMAC")
        f.write(struct.pack("I", 0))  # Version 0 (INT8)

        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                # Simple INT8 quantization
                scale = param.abs().max() / 127
                quantized = (param / scale).round().clamp(-128, 127).to(torch.int8)
                f.write(quantized.numpy().tobytes())

    click.echo("INT8 quantization complete!")


def convert_to_fp16(input_path: str, output_path: str):
    """Convert to FP16 format."""
    click.echo("\nConverting to FP16...")

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise click.ClickException(
            "Missing dependencies. Run: pip install torch transformers"
        )

    model = AutoModelForCausalLM.from_pretrained(
        input_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    # Save as FP16
    model.save_pretrained(output_path)
    click.echo("FP16 conversion complete!")
