"""
Utility modules for the Mojo Gateway.

This package provides:
- Configuration management
- JSON parsing and serialization
- SIMD-accelerated statistics
"""

from .config import Config, get_config
from .json import parse_json, json_get_string, json_get_int, json_get_float, json_get_bool
from .simd_stats import StatsAccumulator, MetricsCollector, RequestMetrics
