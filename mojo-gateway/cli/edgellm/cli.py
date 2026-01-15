"""
EdgeLLM CLI - Main entry point.

Fine-tune, optimize, and deploy custom LLMs to edge devices.
"""

import click
from pathlib import Path
import sys

from . import __version__


@click.group()
@click.version_option(version=__version__, prog_name="edgellm")
def cli():
    """EdgeLLM - Fine-tune, optimize, and deploy LLMs to edge devices.

    Fine-tune once, deploy everywhere - from cloud to edge with deterministic performance.

    Examples:

        # Fine-tune a model (FREE on Google Colab)
        edgellm finetune --base-model smollm-135m --data ./my_dataset.jsonl

        # Quantize to BitNet format
        edgellm quantize --input ./my_model --format bitnet --output ./model.tmac2.bin

        # Run inference server
        edgellm serve --model ./model.tmac2.bin --port 8080

        # Benchmark performance
        edgellm benchmark --model ./model.tmac2.bin
    """
    pass


@cli.command()
@click.option("--base-model", "-b", required=True,
              type=click.Choice(["smollm-135m", "smollm-360m", "qwen2-0.5b", "llama-3.2-1b", "phi-3-mini"]),
              help="Base model to fine-tune")
@click.option("--data", "-d", required=True, type=click.Path(exists=True),
              help="Training data (JSONL, CSV, or HuggingFace dataset)")
@click.option("--output", "-o", default="./output", type=click.Path(),
              help="Output directory for fine-tuned model")
@click.option("--epochs", "-e", default=3, type=int,
              help="Number of training epochs")
@click.option("--batch-size", default=4, type=int,
              help="Training batch size")
@click.option("--learning-rate", "-lr", default=2e-4, type=float,
              help="Learning rate")
@click.option("--lora-r", default=16, type=int,
              help="LoRA rank")
@click.option("--lora-alpha", default=32, type=int,
              help="LoRA alpha")
@click.option("--max-length", default=512, type=int,
              help="Maximum sequence length")
@click.option("--gradient-checkpointing/--no-gradient-checkpointing", default=True,
              help="Enable gradient checkpointing for memory efficiency")
@click.option("--resume", type=click.Path(exists=True),
              help="Resume from checkpoint")
def finetune(base_model, data, output, epochs, batch_size, learning_rate,
             lora_r, lora_alpha, max_length, gradient_checkpointing, resume):
    """Fine-tune a base model on your custom dataset.

    Uses QLoRA for efficient fine-tuning on consumer GPUs or FREE Google Colab.

    Example:
        edgellm finetune --base-model smollm-135m --data ./my_data.jsonl
    """
    click.echo(f"Fine-tuning {base_model} on {data}")
    click.echo(f"Output: {output}")
    click.echo(f"Config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    from .finetune import run_finetune
    run_finetune(
        base_model=base_model,
        data_path=data,
        output_dir=output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_length=max_length,
        gradient_checkpointing=gradient_checkpointing,
        resume_from=resume,
    )


@cli.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True),
              help="Input model directory or HuggingFace model ID")
@click.option("--format", "-f", default="bitnet",
              type=click.Choice(["bitnet", "int4", "int8", "fp16"]),
              help="Quantization format")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output file path (e.g., model.tmac2.bin)")
@click.option("--calibration-data", type=click.Path(exists=True),
              help="Calibration data for quantization (optional)")
@click.option("--num-samples", default=128, type=int,
              help="Number of calibration samples")
def quantize(input, format, output, calibration_data, num_samples):
    """Quantize a model to BitNet 1.58-bit or other formats.

    BitNet 1.58-bit provides 10x smaller models with minimal quality loss.

    Example:
        edgellm quantize --input ./my_model --format bitnet --output ./model.tmac2.bin
    """
    click.echo(f"Quantizing {input} to {format}")
    click.echo(f"Output: {output}")

    from .quantize import run_quantize
    run_quantize(
        input_path=input,
        output_path=output,
        format=format,
        calibration_data=calibration_data,
        num_samples=num_samples,
    )


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True),
              help="Model file (e.g., model.tmac2.bin)")
@click.option("--host", "-h", default="0.0.0.0",
              help="Host to bind to")
@click.option("--port", "-p", default=8080, type=int,
              help="Port to listen on")
@click.option("--workers", "-w", default=1, type=int,
              help="Number of worker processes")
@click.option("--max-batch-size", default=1, type=int,
              help="Maximum batch size for inference")
@click.option("--context-length", default=512, type=int,
              help="Maximum context length")
def serve(model, host, port, workers, max_batch_size, context_length):
    """Start the inference server with OpenAI-compatible API.

    Endpoints:
        POST /v1/chat/completions
        POST /v1/completions
        GET  /v1/models
        GET  /health

    Example:
        edgellm serve --model ./model.tmac2.bin --port 8080
    """
    click.echo(f"Starting EdgeLLM server on {host}:{port}")
    click.echo(f"Model: {model}")

    from .serve import run_server
    run_server(
        model_path=model,
        host=host,
        port=port,
        workers=workers,
        max_batch_size=max_batch_size,
        context_length=context_length,
    )


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True),
              help="Model file (e.g., model.tmac2.bin)")
@click.option("--prompt", "-p", default="Hello, world!",
              help="Input prompt")
@click.option("--max-tokens", default=128, type=int,
              help="Maximum tokens to generate")
@click.option("--temperature", "-t", default=0.7, type=float,
              help="Sampling temperature")
@click.option("--top-p", default=0.9, type=float,
              help="Top-p (nucleus) sampling")
@click.option("--top-k", default=40, type=int,
              help="Top-k sampling")
@click.option("--stream/--no-stream", default=True,
              help="Stream output tokens")
def generate(model, prompt, max_tokens, temperature, top_p, top_k, stream):
    """Generate text from a prompt.

    Example:
        edgellm generate --model ./model.tmac2.bin --prompt "Once upon a time"
    """
    from .generate import run_generate
    run_generate(
        model_path=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stream=stream,
    )


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True),
              help="Model file (e.g., model.tmac2.bin)")
@click.option("--iterations", "-n", default=100, type=int,
              help="Number of iterations")
@click.option("--warmup", default=10, type=int,
              help="Warmup iterations")
@click.option("--prompt-length", default=32, type=int,
              help="Input prompt length (tokens)")
@click.option("--generation-length", default=128, type=int,
              help="Generation length (tokens)")
@click.option("--output", "-o", type=click.Path(),
              help="Output file for detailed results (JSON)")
def benchmark(model, iterations, warmup, prompt_length, generation_length, output):
    """Benchmark model performance.

    Measures:
        - Throughput (tokens/second)
        - Latency (P50, P95, P99)
        - Memory usage
        - First token latency

    Example:
        edgellm benchmark --model ./model.tmac2.bin --iterations 100
    """
    click.echo(f"Benchmarking {model}")
    click.echo(f"Iterations: {iterations}, Warmup: {warmup}")

    from .benchmark import run_benchmark
    run_benchmark(
        model_path=model,
        iterations=iterations,
        warmup=warmup,
        prompt_length=prompt_length,
        generation_length=generation_length,
        output_path=output,
    )


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True),
              help="Model file (e.g., model.tmac2.bin)")
def info(model):
    """Display model information.

    Shows model architecture, size, and configuration.
    """
    click.echo(f"Model: {model}")

    from .info import show_model_info
    show_model_info(model)


@cli.command()
def models():
    """List available base models for fine-tuning.

    Supported models:
        - smollm-135m   (135M params, 35MB BitNet)
        - smollm-360m   (360M params, 90MB BitNet)
        - qwen2-0.5b    (500M params, 125MB BitNet)
        - llama-3.2-1b  (1B params, 200MB BitNet)
        - phi-3-mini    (3.8B params, 750MB BitNet)
    """
    models_list = [
        ("smollm-135m", "135M", "35MB", "Pi Zero 2 W", "5-10 tok/s"),
        ("smollm-360m", "360M", "90MB", "Pi Zero 2 W", "3-6 tok/s"),
        ("qwen2-0.5b", "500M", "125MB", "Pi 4", "8-15 tok/s"),
        ("llama-3.2-1b", "1B", "200MB", "Pi 5", "20-40 tok/s"),
        ("phi-3-mini", "3.8B", "750MB", "Jetson/Mac", "10-20 tok/s"),
    ]

    click.echo("\nAvailable base models for fine-tuning:\n")
    click.echo(f"{'Model':<16} {'Params':<8} {'BitNet Size':<12} {'Min Hardware':<16} {'Speed':<12}")
    click.echo("-" * 70)
    for model, params, size, hardware, speed in models_list:
        click.echo(f"{model:<16} {params:<8} {size:<12} {hardware:<16} {speed:<12}")
    click.echo()


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
