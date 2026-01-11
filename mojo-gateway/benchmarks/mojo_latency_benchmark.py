#!/usr/bin/env python3
"""
Mojo Latency Benchmark - Measures real Mojo inference latency on Fly.io

This measures the actual variance in token generation time for our
Mojo T-MAC BitNet implementation.
"""

import time
import statistics
import requests
import json
import sys
from dataclasses import dataclass


FLYIO_URL = "https://bitnet-inference.fly.dev"
NUM_ITERATIONS = 50
WARMUP_ITERATIONS = 5


@dataclass
class LatencyStats:
    name: str
    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    cv: float
    jitter_ms: float
    tokens_per_sec_mean: float
    tokens_per_sec_std: float


def compute_stats(latencies: list, tps_values: list, name: str) -> LatencyStats:
    """Compute comprehensive latency statistics."""
    sorted_lat = sorted(latencies)
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
        p50_ms=sorted_lat[n // 2],
        p95_ms=sorted_lat[int(n * 0.95)],
        p99_ms=sorted_lat[int(n * 0.99)],
        cv=std / mean if mean > 0 else 0,
        jitter_ms=max(latencies) - min(latencies),
        tokens_per_sec_mean=statistics.mean(tps_values) if tps_values else 0,
        tokens_per_sec_std=statistics.stdev(tps_values) if len(tps_values) > 1 else 0,
    )


def check_server_ready(url: str) -> bool:
    """Check if server is ready."""
    try:
        resp = requests.get(f"{url}/health", timeout=10)
        data = resp.json()
        return data.get("server_status") == "ready"
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def benchmark_mojo_inference(url: str, iterations: int, warmup: int, tokens_per_request: int = 1) -> tuple:
    """
    Benchmark Mojo inference latency.

    Returns:
        tuple: (latencies_ms, tokens_per_sec_values)
    """
    latencies = []
    tps_values = []

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for i in range(warmup):
        try:
            requests.post(
                f"{url}/generate",
                json={"num_tokens": tokens_per_request, "temperature": 0.0},
                timeout=60
            )
        except Exception as e:
            print(f"  Warmup {i} failed: {e}")

    # Benchmark
    print(f"  Running benchmark ({iterations} iterations)...")
    for i in range(iterations):
        try:
            start = time.perf_counter_ns()
            resp = requests.post(
                f"{url}/generate",
                json={"num_tokens": tokens_per_request, "temperature": 0.0},
                timeout=120
            )
            end = time.perf_counter_ns()

            if resp.status_code == 200:
                data = resp.json()
                latency_ms = (end - start) / 1_000_000

                # Also capture the server-reported metrics
                server_elapsed = data.get("elapsed_seconds", 0) * 1000  # ms
                tps = data.get("tokens_per_second", 0)

                latencies.append(latency_ms)
                tps_values.append(tps)

                if (i + 1) % 10 == 0:
                    print(f"    [{i+1}/{iterations}] Latency: {latency_ms:.0f}ms (server: {server_elapsed:.0f}ms), TPS: {tps:.2f}")
            else:
                print(f"    [{i+1}/{iterations}] Error: HTTP {resp.status_code}")

        except Exception as e:
            print(f"    [{i+1}/{iterations}] Exception: {e}")

    return latencies, tps_values


def print_stats(stats: LatencyStats):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"  {stats.name}")
    print(f"{'='*60}")
    print(f"  Iterations:         {stats.count}")
    print(f"  Mean latency:       {stats.mean_ms:.1f} ms")
    print(f"  Std Dev:            {stats.std_ms:.1f} ms")
    print(f"  CV (std/mean):      {stats.cv:.4f}  (lower = more consistent)")
    print(f"  Min:                {stats.min_ms:.1f} ms")
    print(f"  Max:                {stats.max_ms:.1f} ms")
    print(f"  Jitter (range):     {stats.jitter_ms:.1f} ms")
    print(f"  P50:                {stats.p50_ms:.1f} ms")
    print(f"  P95:                {stats.p95_ms:.1f} ms")
    print(f"  P99:                {stats.p99_ms:.1f} ms")
    print(f"  Tokens/sec (mean):  {stats.tokens_per_sec_mean:.2f}")
    print(f"  Tokens/sec (std):   {stats.tokens_per_sec_std:.2f}")


def main():
    print("="*70)
    print("  MOJO T-MAC BITNET LATENCY BENCHMARK")
    print("  Real inference on Fly.io")
    print("="*70)
    print(f"\n  Target: {FLYIO_URL}")
    print(f"  Iterations: {NUM_ITERATIONS}")
    print(f"  Warmup: {WARMUP_ITERATIONS}")

    # Check server
    print("\n[1/3] Checking server status...")
    if not check_server_ready(FLYIO_URL):
        print("ERROR: Server not ready. Exiting.")
        sys.exit(1)
    print("  Server is ready!")

    # Benchmark single token generation
    print("\n[2/3] Benchmarking single token generation...")
    single_latencies, single_tps = benchmark_mojo_inference(
        FLYIO_URL, NUM_ITERATIONS, WARMUP_ITERATIONS, tokens_per_request=1
    )
    single_stats = compute_stats(single_latencies, single_tps, "Mojo BitNet - Single Token")
    print_stats(single_stats)

    # Benchmark 5-token generation
    print("\n[3/3] Benchmarking 5-token generation...")
    multi_latencies, multi_tps = benchmark_mojo_inference(
        FLYIO_URL, NUM_ITERATIONS, WARMUP_ITERATIONS, tokens_per_request=5
    )
    multi_stats = compute_stats(multi_latencies, multi_tps, "Mojo BitNet - 5 Tokens")
    print_stats(multi_stats)

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY: Mojo BitNet Latency Consistency")
    print(f"{'='*70}")

    print(f"\n  Single Token Generation:")
    print(f"    CV (consistency): {single_stats.cv:.4f}")
    print(f"    Jitter: {single_stats.jitter_ms:.0f}ms")
    print(f"    P99: {single_stats.p99_ms:.0f}ms")
    print(f"    Throughput: {single_stats.tokens_per_sec_mean:.2f} tok/s")

    print(f"\n  5-Token Generation:")
    print(f"    CV (consistency): {multi_stats.cv:.4f}")
    print(f"    Jitter: {multi_stats.jitter_ms:.0f}ms")
    print(f"    P99: {multi_stats.p99_ms:.0f}ms")
    print(f"    Throughput: {multi_stats.tokens_per_sec_mean:.2f} tok/s")

    # Note about what this demonstrates
    print(f"\n  Key Insight:")
    print(f"    Mojo's ownership model means NO garbage collection pauses.")
    print(f"    The variance you see here is from:")
    print(f"      - Network latency (Fly.io)")
    print(f"      - Load balancing")
    print(f"      - NOT from GC (there is no GC in Mojo)")

    # Save results
    output = {
        "config": {
            "url": FLYIO_URL,
            "iterations": NUM_ITERATIONS,
            "warmup": WARMUP_ITERATIONS,
        },
        "single_token": {
            "mean_ms": single_stats.mean_ms,
            "std_ms": single_stats.std_ms,
            "cv": single_stats.cv,
            "p99_ms": single_stats.p99_ms,
            "jitter_ms": single_stats.jitter_ms,
            "tps_mean": single_stats.tokens_per_sec_mean,
        },
        "five_tokens": {
            "mean_ms": multi_stats.mean_ms,
            "std_ms": multi_stats.std_ms,
            "cv": multi_stats.cv,
            "p99_ms": multi_stats.p99_ms,
            "jitter_ms": multi_stats.jitter_ms,
            "tps_mean": multi_stats.tokens_per_sec_mean,
        },
        "raw_latencies": {
            "single_token": single_latencies,
            "five_tokens": multi_latencies,
        }
    }

    with open("mojo_latency_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: mojo_latency_results.json")


if __name__ == "__main__":
    main()
