"""
EdgeLLM Fine-tuning Module.

Implements QLoRA fine-tuning for efficient training on consumer GPUs.
"""

import click
from pathlib import Path
from typing import Optional


# Model configurations
MODEL_CONFIGS = {
    "smollm-135m": {
        "hf_id": "HuggingFaceTB/SmolLM-135M",
        "hidden_size": 576,
        "num_layers": 6,
        "num_heads": 9,
    },
    "smollm-360m": {
        "hf_id": "HuggingFaceTB/SmolLM-360M",
        "hidden_size": 960,
        "num_layers": 12,
        "num_heads": 15,
    },
    "qwen2-0.5b": {
        "hf_id": "Qwen/Qwen2-0.5B",
        "hidden_size": 896,
        "num_layers": 24,
        "num_heads": 14,
    },
    "llama-3.2-1b": {
        "hf_id": "meta-llama/Llama-3.2-1B",
        "hidden_size": 2048,
        "num_layers": 16,
        "num_heads": 32,
    },
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "hidden_size": 3072,
        "num_layers": 32,
        "num_heads": 32,
    },
}


def load_dataset(data_path: str, max_length: int = 512):
    """Load and preprocess training dataset."""
    from pathlib import Path

    path = Path(data_path)

    if path.suffix == ".jsonl":
        return load_jsonl_dataset(data_path, max_length)
    elif path.suffix == ".csv":
        return load_csv_dataset(data_path, max_length)
    else:
        # Assume HuggingFace dataset
        return load_hf_dataset(data_path, max_length)


def load_jsonl_dataset(path: str, max_length: int):
    """Load JSONL dataset."""
    import json

    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    click.echo(f"Loaded {len(data)} examples from {path}")
    return data


def load_csv_dataset(path: str, max_length: int):
    """Load CSV dataset."""
    import csv

    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    click.echo(f"Loaded {len(data)} examples from {path}")
    return data


def load_hf_dataset(dataset_name: str, max_length: int):
    """Load HuggingFace dataset."""
    try:
        from datasets import load_dataset as hf_load_dataset

        dataset = hf_load_dataset(dataset_name)
        click.echo(f"Loaded dataset {dataset_name}")
        return dataset
    except ImportError:
        raise click.ClickException(
            "HuggingFace datasets not installed. Run: pip install datasets"
        )


def run_finetune(
    base_model: str,
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_length: int = 512,
    gradient_checkpointing: bool = True,
    resume_from: Optional[str] = None,
):
    """Run QLoRA fine-tuning."""
    click.echo("\n" + "=" * 60)
    click.echo("EdgeLLM Fine-tuning")
    click.echo("=" * 60)

    # Check dependencies
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise click.ClickException(
            f"Missing dependency: {e}\n"
            "Install with: pip install torch transformers peft bitsandbytes"
        )

    # Get model config
    if base_model not in MODEL_CONFIGS:
        raise click.ClickException(f"Unknown model: {base_model}")

    model_config = MODEL_CONFIGS[base_model]
    model_id = model_config["hf_id"]

    click.echo(f"\nBase model: {model_id}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Learning rate: {learning_rate}")
    click.echo(f"LoRA rank: {lora_r}")
    click.echo(f"LoRA alpha: {lora_alpha}")

    # Check GPU
    if torch.cuda.is_available():
        device = "cuda"
        click.echo(f"\nGPU: {torch.cuda.get_device_name(0)}")
        click.echo(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        click.echo("\nNo GPU detected. Training will be slow.")

    # Load dataset
    click.echo("\nLoading dataset...")
    dataset = load_dataset(data_path, max_length)

    # Configure 4-bit quantization for QLoRA
    click.echo("\nLoading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    click.echo(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        push_to_hub=False,
        report_to="none",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train
    click.echo("\nStarting training...")
    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save
    click.echo(f"\nSaving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    click.echo("\nFine-tuning complete!")
    click.echo(f"Next step: edgellm quantize --input {output_dir} --format bitnet --output model.tmac2.bin")
