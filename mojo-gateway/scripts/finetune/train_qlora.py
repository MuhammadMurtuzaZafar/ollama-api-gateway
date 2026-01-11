#!/usr/bin/env python3
"""
EdgeLLM QLoRA Fine-tuning Script

Works on FREE Google Colab / Kaggle with T4 GPU.

Usage:
    python train_qlora.py \
        --base-model HuggingFaceTB/SmolLM-135M \
        --data ./dataset.jsonl \
        --output ./output

Requirements:
    pip install torch transformers peft bitsandbytes datasets accelerate trl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


# Supported models for fine-tuning
SUPPORTED_MODELS = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "smollm-360m": "HuggingFaceTB/SmolLM-360M",
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
}


def load_jsonl_dataset(path: str) -> Dataset:
    """Load dataset from JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    return Dataset.from_list(data)


def format_instruction(example: dict, tokenizer) -> str:
    """Format example for instruction fine-tuning."""
    if "instruction" in example and "response" in example:
        # Alpaca format
        if example.get("input"):
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['response']}"
        else:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    elif "prompt" in example and "completion" in example:
        # OpenAI format
        text = f"{example['prompt']}{example['completion']}"
    elif "text" in example:
        # Raw text
        text = example["text"]
    elif "messages" in example:
        # Chat format
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        raise ValueError(f"Unknown data format: {list(example.keys())}")

    return text


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for EdgeLLM")

    # Model arguments
    parser.add_argument(
        "--base-model", "-b",
        type=str,
        required=True,
        help="Base model name or HuggingFace ID"
    )

    # Data arguments
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Training data (JSONL file or HuggingFace dataset)"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Validation data (optional)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    # Output arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory"
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")

    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target modules for LoRA"
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-model-id", type=str, default=None, help="HuggingFace Hub model ID")

    args = parser.parse_args()

    print("=" * 60)
    print("EdgeLLM QLoRA Fine-tuning")
    print("=" * 60)

    # Resolve model name
    if args.base_model in SUPPORTED_MODELS:
        model_id = SUPPORTED_MODELS[args.base_model]
    else:
        model_id = args.base_model

    print(f"\nBase model: {model_id}")
    print(f"Output: {args.output}")

    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nWARNING: No GPU detected. Training will be very slow.")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"\nLoading dataset from {args.data}...")
    data_path = Path(args.data)
    if data_path.suffix == ".jsonl":
        train_dataset = load_jsonl_dataset(args.data)
    else:
        # Try HuggingFace dataset
        train_dataset = load_dataset(args.data, split="train")

    print(f"Training examples: {len(train_dataset)}")

    # Load validation dataset if provided
    val_dataset = None
    if args.val_data:
        val_path = Path(args.val_data)
        if val_path.suffix == ".jsonl":
            val_dataset = load_jsonl_dataset(args.val_data)
        else:
            val_dataset = load_dataset(args.val_data, split="validation")
        print(f"Validation examples: {len(val_dataset)}")

    # Format dataset
    def formatting_func(example):
        return format_instruction(example, tokenizer)

    # Configure 4-bit quantization for QLoRA
    print("\nConfiguring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"\nLoading model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=100 if val_dataset else None,
        fp16=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        report_to="none",
        seed=args.seed,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=args.max_length,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save
    print(f"\nSaving model to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print("=" * 60)
    print(f"\nModel saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Merge LoRA weights: python merge_lora.py --model {args.output}")
    print(f"  2. Quantize to BitNet: edgellm quantize --input {args.output}_merged --format bitnet")
    print(f"  3. Deploy: edgellm serve --model model.tmac2.bin")


if __name__ == "__main__":
    main()
