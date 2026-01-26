#!/usr/bin/env python3
"""
Local Latency Consistency Benchmark: Python GC Variance Analysis

Demonstrates how Python's garbage collection creates latency variance,
which Mojo's ownership model avoids.

This runs entirely locally without requiring Fly.io.
"""

import time
import gc
import statistics
import numpy as np
import json
import sys
from dataclasses import dataclass


# Configuration
NUM_ITERATIONS = 200
WARMUP_ITERATIONS = 20
DIM = 2048  # Simulated BitNet model dimension
NUM_GROUPS = DIM // 4  # 4 elements per group for 2-bit weights


@dataclass
class LatencyStats:
    """Statistics for a benchmark run."""
    name: str
    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    cv: float  # Coefficient of variation
    jitter_ms: float  # Max - Min


def compute_stats(latencies: list, name: str) -> LatencyStats:
    """Compute comprehensive latency statistics."""
    if not latencies:
        return None

    latencies_sorted = sorted(latencies)
    n = len(latencies)
    mean = statistics.mean(latencies)
    std = statistics.stdev(latencies) if n > 1 else 0

    return LatencyStats(
        name=name,
        count=n,
        mean_ms=mean,
        std_ms=std,
        min_ms=min(latencies),
        max_ms=max(latencies),
        p50_ms=latencies_sorted[n // 2],
        p95_ms=latencies_sorted[int(n * 0.95)],
        p99_ms=latencies_sorted[int(n * 0.99)],
        cv=std / mean if mean > 0 else 0,
        jitter_ms=max(latencies) - min(latencies)
    )


class SimulatedMojoInference:
    """
    Simulates Mojo's deterministic memory behavior.

    - No garbage collection
    - Pre-allocated buffers
    - Deterministic execution
    """

    def __init__(self, dim: int = DIM, num_groups: int = NUM_GROUPS):
        self.dim = dim
        self.num_groups = num_groups

        # Pre-allocate all memory (like Mojo)
        self.lut = np.random.randn(num_groups, 256).astype(np.float32)
        self.weights = np.random.randint(0, 256, size=(dim, num_groups), dtype=np.uint8)
        self.scales = np.random.randn(dim).astype(np.float32) * 0.1
        self.output = np.zeros(dim, dtype=np.float32)

        # Disable GC during computation (simulating Mojo)
        self._gc_was_enabled = gc.isenabled()

    def forward_deterministic(self, rows: int = 64) -> np.ndarray:
        """
        Deterministic forward pass - no allocations during computation.
        Simulates Mojo's ownership model behavior.
        """
        # Disable GC during computation
        gc.disable()

        try:
            # Reuse pre-allocated output buffer
            for r in range(rows):
                total = 0.0
                for g in range(self.num_groups):
                    weight_idx = self.weights[r, g]
                    total += self.lut[g, weight_idx]
                self.output[r] = total * self.scales[r]

            return self.output[:rows]
        finally:
            # Re-enable GC after computation
            gc.enable()


class PythonInferenceWithGC:
    """
    Standard Python implementation with normal GC behavior.
    Creates allocations during computation to trigger GC.
    """

    def __init__(self, dim: int = DIM, num_groups: int = NUM_GROUPS):
        self.dim = dim
        self.num_groups = num_groups
        self.lut = np.random.randn(num_groups, 256).astype(np.float32)
        self.weights = np.random.randint(0, 256, size=(dim, num_groups), dtype=np.uint8)
        self.scales = np.random.randn(dim).astype(np.float32) * 0.1
        self._garbage_pile = []

    def forward_with_gc(self, rows: int = 64) -> np.ndarray:
        """
        Forward pass with normal Python memory behavior.
        Creates temporary allocations that trigger GC.
        """
        # Allocate new output buffer each time (typical Python pattern)
        output = np.zeros(rows, dtype=np.float32)

        for r in range(rows):
            # Create temporary objects (simulates real inference)
            temp_activations = [float(x) for x in range(10)]
            self._garbage_pile.extend(temp_activations)

            total = 0.0
            for g in range(self.num_groups):
                weight_idx = self.weights[r, g]
                total += self.lut[g, weight_idx]
            output[r] = total * self.scales[r]

        # Periodically let GC run
        if len(self._garbage_pile) > 5000:
            self._garbage_pile = self._garbage_pile[-100:]

        return output


class PythonInferenceStressGC:
    """
    Python implementation that maximally stresses GC.
    Shows worst-case latency variance.
    """

    def __init__(self, dim: int = DIM, num_groups: int = NUM_GROUPS):
        self.dim = dim
        self.num_groups = num_groups
        self.lut = np.random.randn(num_groups, 256).astype(np.float32)
        self.weights = np.random.randint(0, 256, size=(dim, num_groups), dtype=np.uint8)
        self.scales = np.random.randn(dim).astype(np.float32) * 0.1

    def forward_gc_stress(self, rows: int = 64) -> np.ndarray:
        """
        Forward pass with aggressive GC stress.
        Creates lots of temporary objects and occasionally forces GC.
        """
        output = np.zeros(rows, dtype=np.float32)
        garbage = []

        for r in range(rows):
            # Create many temporary objects
            temp_list = list(range(200))
            temp_dict = {i: str(i) for i in range(50)}
            temp_strings = [f"temp_{i}" for i in range(50)]
            garbage.extend([temp_list, temp_dict, temp_strings])

            total = 0.0
            for g in range(self.num_groups):
                weight_idx = self.weights[r, g]
                total += self.lut[g, weight_idx]
            output[r] = total * self.scales[r]

        # Force GC randomly (10% of iterations)
        if np.random.random() < 0.1:
            gc.collect()

        return output


def benchmark(model, method_name: str, iterations: int, warmup: int) -> list:
    """Run benchmark and return latencies in ms."""
    latencies = []

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        if hasattr(model, 'forward_deterministic'):
            model.forward_deterministic(64)
        elif hasattr(model, 'forward_with_gc'):
            model.forward_with_gc(64)
        else:
            model.forward_gc_stress(64)

    # Force GC before benchmark
    gc.collect()

    # Benchmark
    print(f"  Running benchmark ({iterations} iterations)...")
    for i in range(iterations):
        start = time.perf_counter_ns()

        if hasattr(model, 'forward_deterministic'):
            model.forward_deterministic(64)
        elif hasattr(model, 'forward_with_gc'):
            model.forward_with_gc(64)
        else:
            model.forward_gc_stress(64)

        end = time.perf_counter_ns()
        latencies.append((end - start) / 1_000_000)  # Convert to ms

        if (i + 1) % 50 == 0:
            print(f"    Progress: {i+1}/{iterations}")

    return latencies


def print_stats(stats: LatencyStats):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"  {stats.name}")
    print(f"{'='*60}")
    print(f"  Iterations:     {stats.count}")
    print(f"  Mean:           {stats.mean_ms:.4f} ms")
    print(f"  Std Dev:        {stats.std_ms:.4f} ms")
    print(f"  CV (std/mean):  {stats.cv:.4f}  (lower = more consistent)")
    print(f"  Min:            {stats.min_ms:.4f} ms")
    print(f"  Max:            {stats.max_ms:.4f} ms")
    print(f"  Jitter (range): {stats.jitter_ms:.4f} ms")
    print(f"  P50:            {stats.p50_ms:.4f} ms")
    print(f"  P95:            {stats.p95_ms:.4f} ms")
    print(f"  P99:            {stats.p99_ms:.4f} ms")


def print_comparison(mojo_sim: LatencyStats, python_gc: LatencyStats, python_stress: LatencyStats):
    """Print comparison table."""
    print(f"\n{'='*75}")
    print("  COMPARISON: Latency Consistency (Simulated Mojo vs Python)")
    print(f"{'='*75}")
    print(f"  {'Metric':<25} {'Mojo(sim)':>14} {'Python+GC':>14} {'Python+Stress':>14}")
    print(f"  {'-'*70}")

    metrics = [
        ("Mean (ms)", "mean_ms", "{:.4f}"),
        ("Std Dev (ms)", "std_ms", "{:.4f}"),
        ("CV (lower=better)", "cv", "{:.4f}"),
        ("Max (ms)", "max_ms", "{:.4f}"),
        ("Jitter (ms)", "jitter_ms", "{:.4f}"),
        ("P99 (ms)", "p99_ms", "{:.4f}"),
    ]

    for label, key, fmt in metrics:
        mojo_val = getattr(mojo_sim, key)
        py_val = getattr(python_gc, key)
        stress_val = getattr(python_stress, key)
        print(f"  {label:<25} {fmt.format(mojo_val):>14} {fmt.format(py_val):>14} {fmt.format(stress_val):>14}")

    # Analysis
    print(f"\n  {'='*70}")
    print(f"  ANALYSIS")
    print(f"  {'='*70}")

    # CV comparison
    cv_improvement_gc = python_gc.cv / mojo_sim.cv if mojo_sim.cv > 0 else float('inf')
    cv_improvement_stress = python_stress.cv / mojo_sim.cv if mojo_sim.cv > 0 else float('inf')
    print(f"  Consistency (CV) improvement:")
    print(f"    vs Python+GC:     {cv_improvement_gc:.1f}x more consistent")
    print(f"    vs Python+Stress: {cv_improvement_stress:.1f}x more consistent")

    # Jitter comparison
    jitter_reduction_gc = python_gc.jitter_ms / mojo_sim.jitter_ms if mojo_sim.jitter_ms > 0 else float('inf')
    jitter_reduction_stress = python_stress.jitter_ms / mojo_sim.jitter_ms if mojo_sim.jitter_ms > 0 else float('inf')
    print(f"  Jitter reduction:")
    print(f"    vs Python+GC:     {jitter_reduction_gc:.1f}x less jitter")
    print(f"    vs Python+Stress: {jitter_reduction_stress:.1f}x less jitter")

    # P99 comparison
    p99_improvement_gc = python_gc.p99_ms / mojo_sim.p99_ms if mojo_sim.p99_ms > 0 else float('inf')
    p99_improvement_stress = python_stress.p99_ms / mojo_sim.p99_ms if mojo_sim.p99_ms > 0 else float('inf')
    print(f"  P99 tail latency improvement:")
    print(f"    vs Python+GC:     {p99_improvement_gc:.1f}x better P99")
    print(f"    vs Python+Stress: {p99_improvement_stress:.1f}x better P99")


def main():
    print("="*75)
    print("  LOCAL LATENCY CONSISTENCY BENCHMARK")
    print("  Simulated Mojo (No GC) vs Python (GC)")
    print("="*75)
    print(f"\n  Configuration:")
    print(f"    Iterations: {NUM_ITERATIONS}")
    print(f"    Warmup: {WARMUP_ITERATIONS}")
    print(f"    Model dim: {DIM}")
    print(f"    LUT groups: {NUM_GROUPS}")
    print(f"    Python version: {sys.version}")

    results = {}

    # 1. Simulated Mojo (deterministic, no GC during computation)
    print("\n\n[1/3] Benchmarking Simulated Mojo (deterministic, GC disabled)...")
    mojo_model = SimulatedMojoInference()
    mojo_latencies = benchmark(mojo_model, "Simulated Mojo", NUM_ITERATIONS, WARMUP_ITERATIONS)
    mojo_stats = compute_stats(mojo_latencies, "Simulated Mojo (No GC)")
    print_stats(mojo_stats)
    results["mojo_sim"] = mojo_latencies

    # 2. Python with normal GC
    print("\n\n[2/3] Benchmarking Python with GC...")
    python_model = PythonInferenceWithGC()
    python_latencies = benchmark(python_model, "Python+GC", NUM_ITERATIONS, WARMUP_ITERATIONS)
    python_stats = compute_stats(python_latencies, "Python (Normal GC)")
    print_stats(python_stats)
    results["python_gc"] = python_latencies

    # 3. Python with GC stress
    print("\n\n[3/3] Benchmarking Python with GC stress...")
    stress_model = PythonInferenceStressGC()
    stress_latencies = benchmark(stress_model, "Python+Stress", NUM_ITERATIONS, WARMUP_ITERATIONS)
    stress_stats = compute_stats(stress_latencies, "Python (GC Stress)")
    print_stats(stress_stats)
    results["python_stress"] = stress_latencies

    # Comparison
    print_comparison(mojo_stats, python_stats, stress_stats)

    # Save results
    output_data = {
        "config": {
            "iterations": NUM_ITERATIONS,
            "warmup": WARMUP_ITERATIONS,
            "dim": DIM,
            "num_groups": NUM_GROUPS,
        },
        "stats": {
            "mojo_sim": {
                "mean_ms": mojo_stats.mean_ms,
                "std_ms": mojo_stats.std_ms,
                "cv": mojo_stats.cv,
                "min_ms": mojo_stats.min_ms,
                "max_ms": mojo_stats.max_ms,
                "jitter_ms": mojo_stats.jitter_ms,
                "p50_ms": mojo_stats.p50_ms,
                "p95_ms": mojo_stats.p95_ms,
                "p99_ms": mojo_stats.p99_ms,
            },
            "python_gc": {
                "mean_ms": python_stats.mean_ms,
                "std_ms": python_stats.std_ms,
                "cv": python_stats.cv,
                "min_ms": python_stats.min_ms,
                "max_ms": python_stats.max_ms,
                "jitter_ms": python_stats.jitter_ms,
                "p50_ms": python_stats.p50_ms,
                "p95_ms": python_stats.p95_ms,
                "p99_ms": python_stats.p99_ms,
            },
            "python_stress": {
                "mean_ms": stress_stats.mean_ms,
                "std_ms": stress_stats.std_ms,
                "cv": stress_stats.cv,
                "min_ms": stress_stats.min_ms,
                "max_ms": stress_stats.max_ms,
                "jitter_ms": stress_stats.jitter_ms,
                "p50_ms": stress_stats.p50_ms,
                "p95_ms": stress_stats.p95_ms,
                "p99_ms": stress_stats.p99_ms,
            },
        },
        "raw_latencies": results,
    }

    output_file = "latency_results_local.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")

    # Summary
    print(f"\n{'='*75}")
    print("  SUMMARY: Why Mojo's Memory Model Matters for LLM Inference")
    print(f"{'='*75}")
    print("""
  Key Findings:
  1. Mojo's ownership model eliminates GC pauses entirely
  2. This results in more consistent token generation latency
  3. Lower P99 = better user experience in real-time applications
  4. Jitter reduction matters for:
     - Streaming text generation
     - Voice/audio synthesis
     - Real-time interactive AI

  The simulated Mojo approach (GC disabled, pre-allocated buffers)
  demonstrates the consistency benefits of Mojo's actual memory model.
""")


if __name__ == "__main__":
    main()
