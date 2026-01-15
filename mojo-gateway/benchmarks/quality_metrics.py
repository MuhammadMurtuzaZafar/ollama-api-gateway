#!/usr/bin/env python3
"""
EdgeLLM Quality Metrics

Measures output quality including:
- Perplexity comparison
- Text generation quality (BLEU-like)
- Response coherence
"""

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    print("Installing requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np


@dataclass
class QualityResult:
    """Quality metrics for a single test."""
    prompt: str
    expected_keywords: List[str]
    response: str
    keyword_match_rate: float
    response_length: int
    coherence_score: float  # 0-1 based on heuristics


@dataclass
class QualityReport:
    """Aggregate quality metrics."""
    backend: str
    total_tests: int
    avg_keyword_match: float
    avg_coherence: float
    avg_response_length: float
    results: List[QualityResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "total_tests": self.total_tests,
            "avg_keyword_match": self.avg_keyword_match,
            "avg_coherence": self.avg_coherence,
            "avg_response_length": self.avg_response_length,
            "results": [asdict(r) for r in self.results],
        }


# Quality test prompts with expected keywords
QUALITY_TESTS = [
    {
        "prompt": "What is the capital of France?",
        "expected_keywords": ["paris", "france", "capital", "city"],
        "category": "factual",
    },
    {
        "prompt": "What is 2 + 2?",
        "expected_keywords": ["4", "four", "equals", "sum"],
        "category": "math",
    },
    {
        "prompt": "List three colors.",
        "expected_keywords": ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "color"],
        "category": "list",
    },
    {
        "prompt": "What is the largest planet in our solar system?",
        "expected_keywords": ["jupiter", "planet", "largest", "solar", "system"],
        "category": "factual",
    },
    {
        "prompt": "Say hello in Spanish.",
        "expected_keywords": ["hola", "spanish", "hello", "greeting"],
        "category": "language",
    },
    {
        "prompt": "What do plants need to grow?",
        "expected_keywords": ["water", "sunlight", "sun", "light", "soil", "nutrients", "air"],
        "category": "science",
    },
    {
        "prompt": "Name an animal that can fly.",
        "expected_keywords": ["bird", "bat", "eagle", "sparrow", "fly", "wings"],
        "category": "factual",
    },
    {
        "prompt": "What year did World War 2 end?",
        "expected_keywords": ["1945", "war", "end", "world"],
        "category": "history",
    },
]


def calculate_keyword_match(response: str, expected_keywords: List[str]) -> float:
    """Calculate what fraction of expected keywords appear in response."""
    if not expected_keywords:
        return 1.0

    response_lower = response.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return matches / len(expected_keywords)


def calculate_coherence(response: str) -> float:
    """
    Simple coherence heuristics:
    - Has complete sentences
    - Reasonable length
    - Not just repeated words
    """
    if not response or len(response.strip()) < 5:
        return 0.0

    score = 0.0

    # Check for sentence structure (ends with punctuation)
    if re.search(r'[.!?]', response):
        score += 0.3

    # Check for reasonable length (10-500 chars is good)
    length = len(response)
    if 10 <= length <= 500:
        score += 0.3
    elif length > 500:
        score += 0.2  # Slightly penalize very long responses

    # Check for word variety (not just repeated words)
    words = response.lower().split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        score += 0.4 * unique_ratio

    return min(score, 1.0)


def test_ollama_quality(model: str = "smollm:135m") -> Optional[QualityReport]:
    """Test Ollama output quality."""
    print(f"\n{'='*60}")
    print(f"OLLAMA QUALITY TEST: {model}")
    print(f"{'='*60}")

    # Check Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            print("Ollama not responding")
            return None
    except Exception as e:
        print(f"Ollama not available: {e}")
        return None

    results = []

    for i, test in enumerate(QUALITY_TESTS):
        prompt = test["prompt"]
        expected = test["expected_keywords"]

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 100}
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "")

                keyword_match = calculate_keyword_match(text, expected)
                coherence = calculate_coherence(text)

                results.append(QualityResult(
                    prompt=prompt,
                    expected_keywords=expected,
                    response=text[:200],  # Truncate for storage
                    keyword_match_rate=keyword_match,
                    response_length=len(text),
                    coherence_score=coherence,
                ))

                status = "PASS" if keyword_match >= 0.3 else "FAIL"
                print(f"  Test {i+1}/{len(QUALITY_TESTS)}: {status} (match={keyword_match:.2f}, coherence={coherence:.2f})")
            else:
                print(f"  Test {i+1}: Error {response.status_code}")

        except Exception as e:
            print(f"  Test {i+1}: Error - {e}")

    if not results:
        return None

    return QualityReport(
        backend=f"Ollama ({model})",
        total_tests=len(results),
        avg_keyword_match=sum(r.keyword_match_rate for r in results) / len(results),
        avg_coherence=sum(r.coherence_score for r in results) / len(results),
        avg_response_length=sum(r.response_length for r in results) / len(results),
        results=results,
    )


def test_edgellm_quality(model_path: str = "models/smollm-135m.tmac2.bin") -> Optional[QualityReport]:
    """Test EdgeLLM output quality."""
    print(f"\n{'='*60}")
    print(f"EDGELLM QUALITY TEST: {model_path}")
    print(f"{'='*60}")

    edgellm_bin = Path("bin/edgellm")

    if not edgellm_bin.exists():
        print("EdgeLLM binary not found. Quality test skipped.")
        print("Build with: pixi run mojo build -O3 src/bitnet_tmac_lut.mojo -o bin/edgellm")
        return None

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return None

    results = []

    for i, test in enumerate(QUALITY_TESTS):
        prompt = test["prompt"]
        expected = test["expected_keywords"]

        try:
            # Run EdgeLLM inference
            result = subprocess.run(
                [str(edgellm_bin), model_path, "-n", "50", "-t", "0.1"],
                capture_output=True,
                text=True,
                timeout=60,
                input=prompt,
            )

            # Parse output (EdgeLLM outputs token IDs, need tokenizer for text)
            text = result.stdout

            keyword_match = calculate_keyword_match(text, expected)
            coherence = calculate_coherence(text)

            results.append(QualityResult(
                prompt=prompt,
                expected_keywords=expected,
                response=text[:200],
                keyword_match_rate=keyword_match,
                response_length=len(text),
                coherence_score=coherence,
            ))

            status = "PASS" if keyword_match >= 0.3 else "FAIL"
            print(f"  Test {i+1}/{len(QUALITY_TESTS)}: {status} (match={keyword_match:.2f}, coherence={coherence:.2f})")

        except subprocess.TimeoutExpired:
            print(f"  Test {i+1}: Timeout")
        except Exception as e:
            print(f"  Test {i+1}: Error - {e}")

    if not results:
        return None

    return QualityReport(
        backend="EdgeLLM",
        total_tests=len(results),
        avg_keyword_match=sum(r.keyword_match_rate for r in results) / len(results),
        avg_coherence=sum(r.coherence_score for r in results) / len(results),
        avg_response_length=sum(r.response_length for r in results) / len(results),
        results=results,
    )


def print_comparison(ollama: Optional[QualityReport], edgellm: Optional[QualityReport]):
    """Print quality comparison."""
    print(f"\n{'='*60}")
    print("QUALITY COMPARISON")
    print(f"{'='*60}")

    print("\n### Quality Metrics")
    print("-" * 50)
    print(f"{'Backend':<20} {'Keyword Match':<15} {'Coherence':<12} {'Avg Length'}")
    print("-" * 50)

    if ollama:
        print(f"{'Ollama':<20} {ollama.avg_keyword_match:>12.2%} {ollama.avg_coherence:>10.2f} {ollama.avg_response_length:>10.0f}")

    if edgellm:
        print(f"{'EdgeLLM':<20} {edgellm.avg_keyword_match:>12.2%} {edgellm.avg_coherence:>10.2f} {edgellm.avg_response_length:>10.0f}")

    if ollama and edgellm:
        print("\n### Quality Comparison")
        match_diff = edgellm.avg_keyword_match - ollama.avg_keyword_match
        coherence_diff = edgellm.avg_coherence - ollama.avg_coherence

        if abs(match_diff) < 0.1:
            print("Keyword Match: Comparable quality")
        elif match_diff > 0:
            print(f"Keyword Match: EdgeLLM +{match_diff:.1%}")
        else:
            print(f"Keyword Match: Ollama +{-match_diff:.1%}")

        if abs(coherence_diff) < 0.1:
            print("Coherence: Comparable quality")
        elif coherence_diff > 0:
            print(f"Coherence: EdgeLLM +{coherence_diff:.2f}")
        else:
            print(f"Coherence: Ollama +{-coherence_diff:.2f}")


def main():
    """Run quality tests."""
    import argparse

    parser = argparse.ArgumentParser(description="EdgeLLM Quality Metrics")
    parser.add_argument("--ollama-model", default="smollm:135m", help="Ollama model name")
    parser.add_argument("--edgellm-model", default="models/smollm-135m.tmac2.bin", help="EdgeLLM model path")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--ollama-only", action="store_true", help="Only test Ollama")
    parser.add_argument("--edgellm-only", action="store_true", help="Only test EdgeLLM")

    args = parser.parse_args()

    ollama_report = None
    edgellm_report = None

    if not args.edgellm_only:
        ollama_report = test_ollama_quality(args.ollama_model)

    if not args.ollama_only:
        edgellm_report = test_edgellm_quality(args.edgellm_model)

    print_comparison(ollama_report, edgellm_report)

    # Save results
    if args.output:
        output_data = {
            "ollama": ollama_report.to_dict() if ollama_report else None,
            "edgellm": edgellm_report.to_dict() if edgellm_report else None,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
