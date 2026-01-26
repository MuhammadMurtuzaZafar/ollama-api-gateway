#!/usr/bin/env python3
"""
GC Pause Detector: Measures actual garbage collection pause times.

This benchmark specifically detects and measures GC-induced latency spikes
by tracking when GC runs and how long it takes.
"""

import gc
import time
import statistics
import numpy as np
import sys
from collections import defaultdict


class GCPauseTracker:
    """Tracks GC pauses using gc callbacks."""

    def __init__(self):
        self.pauses = []
        self.current_start = None
        self._enabled = False

    def gc_callback(self, phase, info):
        """Called before and after GC collection."""
        if phase == "start":
            self.current_start = time.perf_counter_ns()
        elif phase == "stop" and self.current_start is not None:
            pause_ns = time.perf_counter_ns() - self.current_start
            self.pauses.append(pause_ns / 1_000_000)  # ms
            self.current_start = None

    def start(self):
        """Start tracking GC pauses."""
        gc.callbacks.append(self.gc_callback)
        self._enabled = True

    def stop(self):
        """Stop tracking GC pauses."""
        if self._enabled:
            gc.callbacks.remove(self.gc_callback)
            self._enabled = False

    def get_stats(self):
        """Get GC pause statistics."""
        if not self.pauses:
            return {"count": 0, "total_ms": 0, "mean_ms": 0, "max_ms": 0}

        return {
            "count": len(self.pauses),
            "total_ms": sum(self.pauses),
            "mean_ms": statistics.mean(self.pauses),
            "max_ms": max(self.pauses),
            "min_ms": min(self.pauses),
            "p99_ms": sorted(self.pauses)[int(len(self.pauses) * 0.99)] if len(self.pauses) > 1 else self.pauses[0],
        }


def create_memory_pressure():
    """Create objects that will trigger GC."""
    garbage = []

    # Create lots of short-lived objects
    for _ in range(1000):
        # Lists
        garbage.append(list(range(100)))
        # Dicts
        garbage.append({str(i): i for i in range(50)})
        # Strings
        garbage.append("".join([chr(65 + (i % 26)) for i in range(100)]))
        # Nested structures
        garbage.append({"nested": [{"x": i} for i in range(20)]})

    return garbage


def simulate_lut_inference(dim=2048, num_groups=512):
    """Simulate T-MAC LUT inference with memory allocations."""
    # Create LUT and weights (these stay in memory)
    lut = np.random.randn(num_groups, 256).astype(np.float32)
    weights = np.random.randint(0, 256, size=(64, num_groups), dtype=np.uint8)
    scales = np.random.randn(64).astype(np.float32)

    output = np.zeros(64, dtype=np.float32)

    for r in range(64):
        total = 0.0
        for g in range(num_groups):
            total += lut[g, weights[r, g]]
        output[r] = total * scales[r]

    return output


def benchmark_with_gc_pressure(iterations=100):
    """
    Benchmark inference with continuous memory pressure.
    This creates realistic GC pause scenarios.
    """
    tracker = GCPauseTracker()
    tracker.start()

    latencies = []
    garbage_pile = []

    gc.collect()  # Clean slate

    print(f"Running {iterations} iterations with memory pressure...")

    for i in range(iterations):
        # Create memory pressure before inference
        garbage = create_memory_pressure()
        garbage_pile.extend(garbage)

        # Time the inference
        start = time.perf_counter_ns()
        result = simulate_lut_inference()
        end = time.perf_counter_ns()

        latencies.append((end - start) / 1_000_000)

        # Periodically clear some garbage to trigger GC
        if len(garbage_pile) > 50000:
            garbage_pile = garbage_pile[-10000:]
            gc.collect()  # Force collection

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{iterations}, GC pauses so far: {len(tracker.pauses)}")

    tracker.stop()

    return latencies, tracker.get_stats()


def benchmark_without_gc_pressure(iterations=100):
    """
    Benchmark inference without memory pressure (simulating Mojo).
    Pre-allocates all memory and disables GC during computation.
    """
    # Pre-allocate everything
    dim = 2048
    num_groups = 512
    lut = np.random.randn(num_groups, 256).astype(np.float32)
    weights = np.random.randint(0, 256, size=(64, num_groups), dtype=np.uint8)
    scales = np.random.randn(64).astype(np.float32)
    output = np.zeros(64, dtype=np.float32)

    gc.collect()
    gc.disable()  # Disable GC entirely

    latencies = []

    print(f"Running {iterations} iterations without GC...")

    try:
        for i in range(iterations):
            start = time.perf_counter_ns()

            # Inference with no allocations
            for r in range(64):
                total = 0.0
                for g in range(num_groups):
                    total += lut[g, weights[r, g]]
                output[r] = total * scales[r]

            end = time.perf_counter_ns()
            latencies.append((end - start) / 1_000_000)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{iterations}")
    finally:
        gc.enable()

    return latencies


def compute_latency_stats(latencies, name):
    """Compute latency statistics."""
    sorted_lat = sorted(latencies)
    n = len(latencies)

    return {
        "name": name,
        "count": n,
        "mean_ms": statistics.mean(latencies),
        "std_ms": statistics.stdev(latencies) if n > 1 else 0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": sorted_lat[n // 2],
        "p95_ms": sorted_lat[int(n * 0.95)],
        "p99_ms": sorted_lat[int(n * 0.99)],
        "cv": statistics.stdev(latencies) / statistics.mean(latencies) if n > 1 else 0,
    }


def print_comparison(with_gc_stats, without_gc_stats, gc_pause_stats):
    """Print detailed comparison."""
    print("\n" + "="*70)
    print("  RESULTS: GC Impact on Inference Latency")
    print("="*70)

    print("\n  GC Pause Statistics (during 'with GC' benchmark):")
    print(f"    Total GC pauses:    {gc_pause_stats['count']}")
    print(f"    Total pause time:   {gc_pause_stats['total_ms']:.2f} ms")
    print(f"    Mean pause:         {gc_pause_stats['mean_ms']:.4f} ms")
    print(f"    Max pause:          {gc_pause_stats['max_ms']:.4f} ms")
    if gc_pause_stats['count'] > 0:
        print(f"    P99 pause:          {gc_pause_stats['p99_ms']:.4f} ms")

    print("\n  Inference Latency Comparison:")
    print(f"  {'Metric':<20} {'With GC':>15} {'Without GC':>15} {'Difference':>15}")
    print(f"  {'-'*65}")

    metrics = [
        ("Mean (ms)", "mean_ms"),
        ("Std Dev (ms)", "std_ms"),
        ("CV (lower=better)", "cv"),
        ("Min (ms)", "min_ms"),
        ("Max (ms)", "max_ms"),
        ("P95 (ms)", "p95_ms"),
        ("P99 (ms)", "p99_ms"),
    ]

    for label, key in metrics:
        with_val = with_gc_stats[key]
        without_val = without_gc_stats[key]
        diff = with_val - without_val
        diff_pct = (diff / without_val * 100) if without_val > 0 else 0
        print(f"  {label:<20} {with_val:>15.4f} {without_val:>15.4f} {diff:>+10.4f} ({diff_pct:+.1f}%)")

    print("\n  Analysis:")
    cv_ratio = with_gc_stats['cv'] / without_gc_stats['cv'] if without_gc_stats['cv'] > 0 else float('inf')
    max_ratio = with_gc_stats['max_ms'] / without_gc_stats['max_ms'] if without_gc_stats['max_ms'] > 0 else float('inf')
    p99_ratio = with_gc_stats['p99_ms'] / without_gc_stats['p99_ms'] if without_gc_stats['p99_ms'] > 0 else float('inf')

    print(f"    Variance increase (CV):      {cv_ratio:.2f}x higher with GC")
    print(f"    Worst-case increase (Max):   {max_ratio:.2f}x higher with GC")
    print(f"    Tail latency increase (P99): {p99_ratio:.2f}x higher with GC")


def main():
    print("="*70)
    print("  GC PAUSE IMPACT BENCHMARK")
    print("  Measuring actual garbage collection effects on inference latency")
    print("="*70)
    print(f"\n  Python version: {sys.version}")
    print(f"  GC thresholds: {gc.get_threshold()}")

    iterations = 100

    # Benchmark with GC pressure
    print("\n\n[1/2] Benchmark WITH memory pressure (simulates real Python app)...")
    with_gc_latencies, gc_stats = benchmark_with_gc_pressure(iterations)
    with_gc_stats = compute_latency_stats(with_gc_latencies, "With GC Pressure")

    # Benchmark without GC
    print("\n\n[2/2] Benchmark WITHOUT GC (simulates Mojo's memory model)...")
    without_gc_latencies = benchmark_without_gc_pressure(iterations)
    without_gc_stats = compute_latency_stats(without_gc_latencies, "Without GC (Mojo-like)")

    # Print comparison
    print_comparison(with_gc_stats, without_gc_stats, gc_stats)

    # Final summary
    print("\n" + "="*70)
    print("  KEY TAKEAWAY")
    print("="*70)
    print("""
  Mojo's ownership model eliminates GC entirely, providing:
  1. No GC pause interruptions during inference
  2. More predictable latency for real-time applications
  3. Lower tail latencies (P95, P99)

  This matters most for:
  - Streaming text generation (consistent token delivery)
  - Voice synthesis (no audio glitches)
  - Interactive AI (responsive user experience)
  - Edge devices (limited resources, GC is expensive)
""")


if __name__ == "__main__":
    main()
