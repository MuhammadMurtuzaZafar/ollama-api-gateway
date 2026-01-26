"""
EdgeLLM Benchmark Module.

Performance benchmarking for inference.
"""

import click
import time
import json
import statistics
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Benchmark result data."""
    model_path: str
    iterations: int
    prompt_length: int
    generation_length: int

    # Throughput
    tokens_per_second: float
    tokens_per_second_std: float

    # Latency
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_std_ms: float

    # First token
    first_token_latency_ms: float

    # Memory
    memory_mb: float
    peak_memory_mb: float


def run_benchmark(
    model_path: str,
    iterations: int = 100,
    warmup: int = 10,
    prompt_length: int = 32,
    generation_length: int = 128,
    output_path: Optional[str] = None,
):
    """Run comprehensive benchmark."""
    click.echo("\n" + "=" * 60)
    click.echo("EdgeLLM Benchmark")
    click.echo("=" * 60)

    click.echo(f"\nModel: {model_path}")
    click.echo(f"Iterations: {iterations}")
    click.echo(f"Warmup: {warmup}")
    click.echo(f"Prompt length: {prompt_length} tokens")
    click.echo(f"Generation length: {generation_length} tokens")

    # Load model
    click.echo("\nLoading model...")
    model = load_model(model_path)

    # Get memory baseline
    import psutil
    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024

    # Generate test prompt
    prompt = generate_test_prompt(prompt_length)

    # Warmup
    click.echo(f"\nWarming up ({warmup} iterations)...")
    for i in range(warmup):
        _ = model.generate(prompt, max_tokens=generation_length)
        click.echo(f"  Warmup {i + 1}/{warmup}", nl=False)
        click.echo("\r", nl=False)
    click.echo("  Warmup complete!      ")

    # Benchmark
    click.echo(f"\nRunning benchmark ({iterations} iterations)...")

    latencies: List[float] = []
    tokens_per_sec: List[float] = []
    first_token_latencies: List[float] = []
    peak_memory = baseline_memory

    for i in range(iterations):
        # Measure generation
        start = time.perf_counter()

        # First token timing
        first_token_time = None
        tokens_generated = 0

        for token in model.generate_stream(prompt, max_tokens=generation_length):
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            tokens_generated += 1

        end = time.perf_counter()
        elapsed = end - start

        # Record metrics
        latencies.append(elapsed * 1000)  # ms
        tokens_per_sec.append(tokens_generated / elapsed)
        if first_token_time:
            first_token_latencies.append(first_token_time * 1000)

        # Check memory
        current_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)

        # Progress
        if (i + 1) % 10 == 0:
            click.echo(f"  Progress: {i + 1}/{iterations}")

    # Calculate statistics
    result = BenchmarkResult(
        model_path=model_path,
        iterations=iterations,
        prompt_length=prompt_length,
        generation_length=generation_length,
        tokens_per_second=statistics.mean(tokens_per_sec),
        tokens_per_second_std=statistics.stdev(tokens_per_sec) if len(tokens_per_sec) > 1 else 0,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        latency_p99_ms=percentile(latencies, 99),
        latency_mean_ms=statistics.mean(latencies),
        latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        first_token_latency_ms=statistics.mean(first_token_latencies) if first_token_latencies else 0,
        memory_mb=process.memory_info().rss / 1024 / 1024 - baseline_memory,
        peak_memory_mb=peak_memory - baseline_memory,
    )

    # Print results
    print_results(result)

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        click.echo(f"\nResults saved to {output_path}")

    return result


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


def print_results(result: BenchmarkResult):
    """Print benchmark results."""
    click.echo("\n" + "=" * 60)
    click.echo("BENCHMARK RESULTS")
    click.echo("=" * 60)

    click.echo("\nThroughput:")
    click.echo(f"  Tokens/second:     {result.tokens_per_second:.1f} +/- {result.tokens_per_second_std:.1f}")

    click.echo("\nLatency (per generation):")
    click.echo(f"  Mean:              {result.latency_mean_ms:.1f} ms +/- {result.latency_std_ms:.1f} ms")
    click.echo(f"  P50:               {result.latency_p50_ms:.1f} ms")
    click.echo(f"  P95:               {result.latency_p95_ms:.1f} ms")
    click.echo(f"  P99:               {result.latency_p99_ms:.1f} ms")

    click.echo("\nFirst Token Latency:")
    click.echo(f"  Mean:              {result.first_token_latency_ms:.1f} ms")

    click.echo("\nMemory:")
    click.echo(f"  Model memory:      {result.memory_mb:.1f} MB")
    click.echo(f"  Peak memory:       {result.peak_memory_mb:.1f} MB")

    # Performance grade
    click.echo("\nPerformance Grade:")
    if result.tokens_per_second >= 40:
        grade = "EXCELLENT"
        color = "green"
    elif result.tokens_per_second >= 20:
        grade = "GOOD"
        color = "green"
    elif result.tokens_per_second >= 10:
        grade = "ACCEPTABLE"
        color = "yellow"
    else:
        grade = "NEEDS IMPROVEMENT"
        color = "red"

    click.secho(f"  {grade} ({result.tokens_per_second:.1f} tok/s)", fg=color)

    click.echo("\n" + "=" * 60)


def generate_test_prompt(length: int) -> str:
    """Generate a test prompt of specified token length."""
    # Approximate: 1 word ~ 1.3 tokens
    words_needed = int(length / 1.3)
    test_words = [
        "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
        "and", "then", "runs", "away", "into", "the", "forest", "where", "it",
        "finds", "a", "cozy", "den", "to", "rest", "for", "the", "night",
    ]
    prompt_words = []
    for i in range(words_needed):
        prompt_words.append(test_words[i % len(test_words)])
    return " ".join(prompt_words)


def load_model(model_path: str):
    """Load the T-MAC model."""
    # TODO: Implement actual model loading with C FFI runtime

    class MockModel:
        def generate(
            self,
            prompt: str,
            max_tokens: int = 128,
            temperature: float = 0.7,
            top_p: float = 0.9,
        ) -> str:
            return "mock response " * max_tokens

        def generate_stream(
            self,
            prompt: str,
            max_tokens: int = 128,
            temperature: float = 0.7,
            top_p: float = 0.9,
        ):
            import time
            for i in range(max_tokens):
                yield "token"
                time.sleep(0.001)  # ~1000 tok/s mock

    return MockModel()
