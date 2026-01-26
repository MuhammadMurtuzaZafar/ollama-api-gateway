"""
EdgeLLM Info Module.

Display model information.
"""

import click
import struct
from pathlib import Path


def show_model_info(model_path: str):
    """Display model information."""
    click.echo("\n" + "=" * 60)
    click.echo("EdgeLLM Model Information")
    click.echo("=" * 60)

    path = Path(model_path)

    if not path.exists():
        raise click.ClickException(f"Model not found: {model_path}")

    # File info
    click.echo(f"\nFile: {path.absolute()}")
    click.echo(f"Size: {path.stat().st_size / 1024 / 1024:.1f} MB")

    # Try to read T-MAC header
    with open(model_path, "rb") as f:
        magic = f.read(4)

        if magic == b"TMAC":
            show_tmac_info(f, path)
        elif magic == b"GGUF":
            click.echo("\nFormat: GGUF (llama.cpp format)")
            click.echo("Note: Convert to T-MAC format for EdgeLLM")
        else:
            # Try HuggingFace format
            config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
            if config_path.exists():
                show_huggingface_info(config_path)
            else:
                click.echo("\nFormat: Unknown")
                click.echo("Supported formats: T-MAC (.tmac2.bin), HuggingFace")


def show_tmac_info(f, path: Path):
    """Show T-MAC format model info."""
    click.echo("\nFormat: T-MAC (EdgeLLM native)")

    # Read version
    version = struct.unpack("I", f.read(4))[0]
    click.echo(f"Version: {version}")

    if version >= 2:
        # Read model config
        hidden_size = struct.unpack("I", f.read(4))[0]
        num_layers = struct.unpack("I", f.read(4))[0]
        num_heads = struct.unpack("I", f.read(4))[0]
        vocab_size = struct.unpack("I", f.read(4))[0]

        # Quantization config
        bits = struct.unpack("I", f.read(4))[0]
        group_size = struct.unpack("I", f.read(4))[0]

        click.echo("\nModel Configuration:")
        click.echo(f"  Hidden size:    {hidden_size}")
        click.echo(f"  Layers:         {num_layers}")
        click.echo(f"  Attention heads:{num_heads}")
        click.echo(f"  Vocabulary:     {vocab_size:,}")

        click.echo("\nQuantization:")
        if bits == 2:
            click.echo("  Format:         BitNet 1.58-bit")
        elif bits == 4:
            click.echo("  Format:         INT4")
        elif bits == 8:
            click.echo("  Format:         INT8")
        else:
            click.echo(f"  Format:         {bits}-bit")
        click.echo(f"  Group size:     {group_size}")

        # Estimate model size
        params = estimate_parameters(hidden_size, num_layers, num_heads, vocab_size)
        click.echo("\nModel Size:")
        click.echo(f"  Parameters:     {params:,}")
        click.echo(f"  Original (FP16):{params * 2 / 1024 / 1024:.1f} MB")
        click.echo(f"  Compressed:     {path.stat().st_size / 1024 / 1024:.1f} MB")
        click.echo(f"  Compression:    {(params * 2) / path.stat().st_size:.1f}x")

        # Recommend hardware
        click.echo("\nRecommended Hardware:")
        file_mb = path.stat().st_size / 1024 / 1024
        if file_mb < 50:
            click.echo("  Raspberry Pi Zero 2 W (512MB RAM)")
            click.echo("  Expected: 5-10 tok/s")
        elif file_mb < 150:
            click.echo("  Raspberry Pi 4 (4GB RAM)")
            click.echo("  Expected: 10-20 tok/s")
        elif file_mb < 300:
            click.echo("  Raspberry Pi 5 (8GB RAM)")
            click.echo("  Expected: 20-40 tok/s")
        else:
            click.echo("  Jetson Nano / Mac M1/M2")
            click.echo("  Expected: 15-30 tok/s")


def show_huggingface_info(config_path: Path):
    """Show HuggingFace format model info."""
    import json

    click.echo("\nFormat: HuggingFace (transformers)")

    with open(config_path) as f:
        config = json.load(f)

    click.echo("\nModel Configuration:")
    if "hidden_size" in config:
        click.echo(f"  Hidden size:     {config['hidden_size']}")
    if "num_hidden_layers" in config:
        click.echo(f"  Layers:          {config['num_hidden_layers']}")
    if "num_attention_heads" in config:
        click.echo(f"  Attention heads: {config['num_attention_heads']}")
    if "vocab_size" in config:
        click.echo(f"  Vocabulary:      {config['vocab_size']:,}")
    if "model_type" in config:
        click.echo(f"  Architecture:    {config['model_type']}")

    click.echo("\nNote: Convert to T-MAC format for EdgeLLM deployment:")
    click.echo(f"  edgellm quantize --input {config_path.parent} --format bitnet --output model.tmac2.bin")


def estimate_parameters(hidden_size: int, num_layers: int, num_heads: int, vocab_size: int) -> int:
    """Estimate number of parameters."""
    # Embedding
    embed_params = vocab_size * hidden_size

    # Per layer (approximate Llama-style):
    # - Q, K, V projections: 3 * hidden_size * hidden_size
    # - Output projection: hidden_size * hidden_size
    # - MLP (up/gate/down): ~3 * hidden_size * 4 * hidden_size
    # - Norms: 2 * hidden_size
    per_layer = 4 * hidden_size * hidden_size + 12 * hidden_size * hidden_size + 2 * hidden_size

    # Total
    total = embed_params + num_layers * per_layer + hidden_size  # Final norm

    return total
