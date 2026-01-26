#!/usr/bin/env python3
"""
EdgeLLM LoRA Merge Script

Merges LoRA adapter weights back into the base model.
This creates a standalone model ready for quantization.

Usage:
    python merge_lora.py \
        --model ./finetuned_lora \
        --output ./merged_model

Requirements:
    pip install torch transformers peft
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to fine-tuned LoRA model"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for merged model (default: <model>_merged)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model ID (auto-detected from adapter config if not provided)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Output dtype"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EdgeLLM LoRA Merge")
    print("=" * 60)

    model_path = Path(args.model)
    output_path = args.output or str(model_path) + "_merged"

    print(f"\nInput: {model_path}")
    print(f"Output: {output_path}")

    # Load adapter config to get base model
    print("\nLoading adapter configuration...")
    config = PeftConfig.from_pretrained(args.model)

    base_model_id = args.base_model or config.base_model_name_or_path
    print(f"Base model: {base_model_id}")

    # Select dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    print(f"Output dtype: {args.dtype}")

    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="cpu",  # Use CPU to avoid GPU memory issues
        trust_remote_code=True,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load LoRA model
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.model)

    # Merge weights
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()

    # Calculate sizes
    base_params = sum(p.numel() for p in base_model.parameters())
    merged_params = sum(p.numel() for p in merged_model.parameters())

    print(f"\nBase model parameters: {base_params:,}")
    print(f"Merged model parameters: {merged_params:,}")

    # Save merged model
    print(f"\nSaving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Calculate output size
    output_dir = Path(output_path)
    total_size = sum(f.stat().st_size for f in output_dir.glob("**/*") if f.is_file())

    print(f"\nOutput size: {total_size / 1024 / 1024:.1f} MB")

    # Push to hub if requested
    if args.push_to_hub and args.hub_model_id:
        print(f"\nPushing to HuggingFace Hub: {args.hub_model_id}")
        merged_model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)

    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)
    print(f"\nMerged model saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  Quantize to BitNet: edgellm quantize --input {output_path} --format bitnet --output model.tmac2.bin")


if __name__ == "__main__":
    main()
