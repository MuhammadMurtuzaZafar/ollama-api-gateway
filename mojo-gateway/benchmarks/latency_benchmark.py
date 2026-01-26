#!/usr/bin/env python3
"""
Latency Consistency Benchmark: Mojo vs Python

Measures timing variance to demonstrate Mojo's deterministic memory model
(no GC pauses) vs Python's garbage collection.

Metrics:
- Mean latency
- Standard deviation
- P50, P95, P99 latencies
- Max latency (worst case GC pause indicator)
- Coefficient of variation (CV = std/mean)
"""

import time
import gc
import statistics
import numpy as np
from typing import List, Tuple
import struct
import os
import subprocess
import requests
import json

# Configuration
NUM_ITERATIONS = 100
WARMUP_ITERATIONS = 10
DIM = 2048  # Simulated model dimension
NUM_GROUPS = DIM // 4  # 4 elements per group for 2-bit weights


class PythonLUTInference:
    """
    Python implementation of T-MAC LUT inference.
    Subject to GC pauses.
    """

    def __init__(self, dim: int = DIM, num_groups: int = NUM_GROUPS):
        self.dim = dim
        self.num_groups = num_groups

        # Pre-compute LUT (256 entries per group, 4 activations combined)
        # Each entry is the sum of 4 ternary weight * activation products
        self.lut = np.random.randn(num_groups, 256).astype(np.float32)

        # Simulated packed weights (1 byte = 4 ternary weights)
        self.weights = np.random.randint(0, 256, size=(dim, num_groups), dtype=np.uint8)

        # Scales
        self.scales = np.random.randn(dim).astype(np.float32) * 0.1

        # Create some garbage to trigger GC
        self._garbage = []

    def forward_single_row(self, row: int) -> float:
        """Compute single output using LUT lookup."""
        total = 0.0
        for g in range(self.num_groups):
            weight_idx = self.weights[row, g]
            total += self.lut[g, weight_idx]
        return total * self.scales[row]

    def forward_batch(self, rows: int = 64) -> np.ndarray:
        """Compute multiple rows (simulates token generation)."""
        output = np.zeros(rows, dtype=np.float32)

        # Create temporary allocations to stress GC
        temp_data = [np.random.randn(100) for _ in range(10)]
        self._garbage.extend(temp_data)

        for r in range(rows):
            output[r] = self.forward_single_row(r)

        # Periodically clear garbage to trigger GC
        if len(self._garbage) > 1000:
            self._garbage = self._garbage[-100:]

        return output


class PythonLUTInferenceWithGC:
    """
    Python implementation that explicitly triggers GC to show worst case.
    """

    def __init__(self, dim: int = DIM, num_groups: int = NUM_GROUPS):
        self.dim = dim
        self.num_groups = num_groups
        self.lut = np.random.randn(num_groups, 256).astype(np.float32)
        self.weights = np.random.randint(0, 256, size=(dim, num_groups), dtype=np.uint8)
        self.scales = np.random.randn(dim).astype(np.float32) * 0.1

    def forward_with_gc_pressure(self, rows: int = 64) -> np.ndarray:
        """Forward pass with GC pressure."""
        output = np.zeros(rows, dtype=np.float32)

        # Create garbage
        garbage = []
        for r in range(rows):
            # Allocate temporary objects
            temp = [list(range(100)) for _ in range(5)]
            garbage.extend(temp)

            total = 0.0
            for g in range(self.num_groups):
                weight_idx = self.weights[r, g]
                total += self.lut[g, weight_idx]
            output[r] = total * self.scales[r]

        # Force GC occasionally
        if np.random.random() < 0.1:
            gc.collect()

        return output


def benchmark_python_inference(model: PythonLUTInference,
                                num_iterations: int,
                                warmup: int = 10) -> List[float]:
    """Benchmark Python inference latency."""
    latencies = []

    # Warmup
    for _ in range(warmup):
        model.forward_batch(64)

    # Benchmark
    for i in range(num_iterations):
        start = time.perf_counter_ns()
        model.forward_batch(64)
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1_000_000)  # Convert to ms

    return latencies


def benchmark_python_with_gc(model: PythonLUTInferenceWithGC,
                              num_iterations: int,
                              warmup: int = 10) -> List[float]:
    """Benchmark Python with GC pressure."""
    latencies = []

    # Warmup
    for _ in range(warmup):
        model.forward_with_gc_pressure(64)

    # Benchmark
    for i in range(num_iterations):
        start = time.perf_counter_ns()
        model.forward_with_gc_pressure(64)
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1_000_000)  # Convert to ms

    return latencies


def benchmark_flyio_mojo(url: str, num_iterations: int, warmup: int = 10) -> List[float]:
    """Benchmark Mojo implementation via Fly.io API."""
    latencies = []

    # Check if server is available
    try:
        health = requests.get(f"{url}/health", timeout=10)
        if health.status_code != 200:
            print(f"Fly.io server unhealthy: {health.text}")
            return []
    except Exception as e:
        print(f"Cannot connect to Fly.io: {e}")
        return []

    # Warmup
    print("Warming up Fly.io server...")
    for _ in range(warmup):
        try:
            requests.post(f"{url}/generate",
                         json={"num_tokens": 1, "temperature": 0.0},
                         timeout=60)
        except:
            pass

    # Benchmark
    print(f"Running {num_iterations} iterations...")
    for i in range(num_iterations):
        start = time.perf_counter_ns()
        try:
            response = requests.post(
                f"{url}/generate",
                json={"num_tokens": 1, "temperature": 0.0},
                timeout=120
            )
            if response.status_code == 200:
                end = time.perf_counter_ns()
                latencies.append((end - start) / 1_000_000)  # ms

                if (i + 1) % 10 == 0:
                    print(f"  Iteration {i+1}/{num_iterations}: {latencies[-1]:.2f}ms")
            else:
                print(f"  Error at iteration {i}: {response.status_code}")
        except Exception as e:
            print(f"  Exception at iteration {i}: {e}")

    return latencies


def compute_stats(latencies: List[float], name: str) -> dict:
    """Compute comprehensive latency statistics."""
    if not latencies:
        return {"name": name, "error": "No data"}

    latencies_sorted = sorted(latencies)
    n = len(latencies)

    stats = {
        "name": name,
        "count": n,
        "mean_ms": statistics.mean(latencies),
        "std_ms": statistics.stdev(latencies) if n > 1 else 0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": latencies_sorted[n // 2],
        "p95_ms": latencies_sorted[int(n * 0.95)],
        "p99_ms": latencies_sorted[int(n * 0.99)],
    }

    # Coefficient of variation (lower = more consistent)
    stats["cv"] = stats["std_ms"] / stats["mean_ms"] if stats["mean_ms"] > 0 else 0

    # Jitter (max - min, indicator of worst-case variance)
    stats["jitter_ms"] = stats["max_ms"] - stats["min_ms"]

    return stats


def print_stats(stats: dict):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"  {stats['name']}")
    print(f"{'='*60}")

    if "error" in stats:
        print(f"  Error: {stats['error']}")
        return

    print(f"  Iterations:     {stats['count']}")
    print(f"  Mean:           {stats['mean_ms']:.3f} ms")
    print(f"  Std Dev:        {stats['std_ms']:.3f} ms")
    print(f"  CV (std/mean):  {stats['cv']:.4f}  (lower = more consistent)")
    print(f"  Min:            {stats['min_ms']:.3f} ms")
    print(f"  Max:            {stats['max_ms']:.3f} ms")
    print(f"  Jitter (range): {stats['jitter_ms']:.3f} ms")
    print(f"  P50:            {stats['p50_ms']:.3f} ms")
    print(f"  P95:            {stats['p95_ms']:.3f} ms")
    print(f"  P99:            {stats['p99_ms']:.3f} ms")


def print_comparison(mojo_stats: dict, python_stats: dict, python_gc_stats: dict):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print("  COMPARISON: Latency Consistency")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'Mojo':>12} {'Python':>12} {'Python+GC':>12}")
    print(f"  {'-'*56}")

    metrics = [
        ("Mean (ms)", "mean_ms"),
        ("Std Dev (ms)", "std_ms"),
        ("CV (lower=better)", "cv"),
        ("Max (ms)", "max_ms"),
        ("Jitter (ms)", "jitter_ms"),
        ("P99 (ms)", "p99_ms"),
    ]

    for label, key in metrics:
        mojo_val = mojo_stats.get(key, "N/A")
        py_val = python_stats.get(key, "N/A")
        py_gc_val = python_gc_stats.get(key, "N/A")

        if isinstance(mojo_val, float):
            print(f"  {label:<20} {mojo_val:>12.3f} {py_val:>12.3f} {py_gc_val:>12.3f}")
        else:
            print(f"  {label:<20} {mojo_val:>12} {py_val:>12} {py_gc_val:>12}")

    # Consistency winner
    print(f"\n  Consistency Analysis:")
    if mojo_stats.get("cv", float('inf')) < python_stats.get("cv", float('inf')):
        improvement = python_stats["cv"] / mojo_stats["cv"] if mojo_stats["cv"] > 0 else float('inf')
        print(f"  - Mojo is {improvement:.1f}x more consistent than Python (by CV)")
    else:
        print(f"  - Python is more consistent (unexpected)")

    if mojo_stats.get("jitter_ms", float('inf')) < python_stats.get("jitter_ms", float('inf')):
        reduction = python_stats["jitter_ms"] / mojo_stats["jitter_ms"] if mojo_stats["jitter_ms"] > 0 else float('inf')
        print(f"  - Mojo has {reduction:.1f}x less jitter than Python")


def main():
    print("="*70)
    print("  LATENCY CONSISTENCY BENCHMARK")
    print("  Mojo (No GC) vs Python (GC)")
    print("="*70)
    print(f"\n  Configuration:")
    print(f"    Iterations: {NUM_ITERATIONS}")
    print(f"    Warmup: {WARMUP_ITERATIONS}")
    print(f"    Model dim: {DIM}")
    print(f"    LUT groups: {NUM_GROUPS}")

    results = {}

    # 1. Python baseline
    print("\n\n[1/3] Benchmarking Python LUT inference...")
    python_model = PythonLUTInference()
    python_latencies = benchmark_python_inference(
        python_model, NUM_ITERATIONS, WARMUP_ITERATIONS
    )
    results["python"] = compute_stats(python_latencies, "Python LUT (baseline)")
    print_stats(results["python"])

    # 2. Python with GC pressure
    print("\n\n[2/3] Benchmarking Python with GC pressure...")
    python_gc_model = PythonLUTInferenceWithGC()
    python_gc_latencies = benchmark_python_with_gc(
        python_gc_model, NUM_ITERATIONS, WARMUP_ITERATIONS
    )
    results["python_gc"] = compute_stats(python_gc_latencies, "Python LUT (GC stress)")
    print_stats(results["python_gc"])

    # 3. Mojo via Fly.io
    print("\n\n[3/3] Benchmarking Mojo via Fly.io...")
    flyio_url = os.environ.get("FLYIO_URL", "https://bitnet-inference.fly.dev")
    mojo_latencies = benchmark_flyio_mojo(flyio_url, NUM_ITERATIONS, WARMUP_ITERATIONS)
    results["mojo"] = compute_stats(mojo_latencies, f"Mojo T-MAC ({flyio_url})")
    print_stats(results["mojo"])

    # Comparison
    print_comparison(results["mojo"], results["python"], results["python_gc"])

    # Save raw data
    output_file = "latency_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "iterations": NUM_ITERATIONS,
                "warmup": WARMUP_ITERATIONS,
                "dim": DIM,
                "num_groups": NUM_GROUPS,
            },
            "results": results,
            "raw_latencies": {
                "python": python_latencies,
                "python_gc": python_gc_latencies,
                "mojo": mojo_latencies,
            }
        }, f, indent=2)
    print(f"\n\nRaw data saved to: {output_file}")


if __name__ == "__main__":
    main()
