"""
EdgeLLM CLI - Fine-tune, optimize, and deploy LLMs to edge devices.

Usage:
    edgellm finetune --base-model smollm-135m --data ./data.jsonl
    edgellm quantize --input ./model --format bitnet --output ./model.tmac2.bin
    edgellm serve --model ./model.tmac2.bin --port 8080
    edgellm benchmark --model ./model.tmac2.bin
"""

__version__ = "0.1.0"
__author__ = "EdgeLLM Team"
