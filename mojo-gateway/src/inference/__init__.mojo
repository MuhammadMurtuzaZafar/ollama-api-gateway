"""
Inference Engine module.

This package provides integration with MAX Engine for
high-performance LLM inference.
"""

from .max_engine import (
    MAXInferenceEngine,
    InferenceConfig,
    InferenceResult,
    GenerationParams,
    get_inference_engine,
    initialize_inference_engine
)
