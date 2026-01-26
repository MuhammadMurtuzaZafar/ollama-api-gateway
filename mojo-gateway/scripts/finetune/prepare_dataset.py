#!/usr/bin/env python3
"""
EdgeLLM Dataset Preparation Script

Converts various data formats to the JSONL format expected by train_qlora.py.

Supported input formats:
    - CSV with 'instruction' and 'response' columns
    - JSON array of instruction/response pairs
    - Text file with delimiter-separated pairs
    - HuggingFace dataset

Output format (JSONL):
    {"instruction": "...", "response": "..."}
    {"instruction": "...", "input": "...", "response": "..."}

Usage:
    python prepare_dataset.py \
        --input ./raw_data.csv \
        --output ./dataset.jsonl \
        --format alpaca
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any


def load_csv(path: str) -> List[Dict[str, Any]]:
    """Load data from CSV file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    return data


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_text(path: str, delimiter: str = "\n---\n") -> List[Dict[str, Any]]:
    """Load data from text file with delimited pairs."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    pairs = content.split(delimiter)
    for pair in pairs:
        parts = pair.strip().split("\n\n", 1)
        if len(parts) == 2:
            data.append({
                "instruction": parts[0].strip(),
                "response": parts[1].strip(),
            })

    return data


def convert_to_alpaca(data: List[Dict[str, Any]], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """Convert data to Alpaca format (instruction/input/response)."""
    converted = []
    for item in data:
        new_item = {}

        # Map instruction
        if "instruction" in mapping:
            new_item["instruction"] = item.get(mapping["instruction"], "")
        elif "instruction" in item:
            new_item["instruction"] = item["instruction"]
        elif "question" in item:
            new_item["instruction"] = item["question"]
        elif "prompt" in item:
            new_item["instruction"] = item["prompt"]
        else:
            continue  # Skip if no instruction found

        # Map input (optional)
        if "input" in mapping and mapping["input"] in item:
            new_item["input"] = item[mapping["input"]]
        elif "input" in item:
            new_item["input"] = item["input"]
        elif "context" in item:
            new_item["input"] = item["context"]

        # Map response
        if "response" in mapping:
            new_item["response"] = item.get(mapping["response"], "")
        elif "response" in item:
            new_item["response"] = item["response"]
        elif "answer" in item:
            new_item["response"] = item["answer"]
        elif "completion" in item:
            new_item["response"] = item["completion"]
        elif "output" in item:
            new_item["response"] = item["output"]
        else:
            continue  # Skip if no response found

        converted.append(new_item)

    return converted


def convert_to_chat(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert data to chat format (messages array)."""
    converted = []
    for item in data:
        messages = []

        # Handle existing messages
        if "messages" in item:
            converted.append(item)
            continue

        # System message (optional)
        if "system" in item:
            messages.append({"role": "system", "content": item["system"]})

        # User message
        if "instruction" in item:
            user_content = item["instruction"]
            if "input" in item and item["input"]:
                user_content += f"\n\n{item['input']}"
            messages.append({"role": "user", "content": user_content})
        elif "prompt" in item:
            messages.append({"role": "user", "content": item["prompt"]})

        # Assistant response
        if "response" in item:
            messages.append({"role": "assistant", "content": item["response"]})
        elif "completion" in item:
            messages.append({"role": "assistant", "content": item["completion"]})

        if len(messages) >= 2:  # At least user + assistant
            converted.append({"messages": messages})

    return converted


def save_jsonl(data: List[Dict[str, Any]], path: str):
    """Save data to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_dataset(
    data: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """Split dataset into train and validation."""
    import random
    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - val_ratio))
    return data[:split_idx], data[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for EdgeLLM fine-tuning")

    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--format", "-f",
        choices=["alpaca", "chat", "auto"],
        default="alpaca",
        help="Output format"
    )
    parser.add_argument(
        "--input-format",
        choices=["csv", "json", "jsonl", "text", "auto"],
        default="auto",
        help="Input file format"
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default=None,
        help="Output path for validation set"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--instruction-col",
        type=str,
        default=None,
        help="Column name for instruction"
    )
    parser.add_argument(
        "--response-col",
        type=str,
        default=None,
        help="Column name for response"
    )
    parser.add_argument(
        "--input-col",
        type=str,
        default=None,
        help="Column name for input context (optional)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 60)
    print("EdgeLLM Dataset Preparation")
    print("=" * 60)

    input_path = Path(args.input)
    print(f"\nInput: {input_path}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")

    # Detect input format
    if args.input_format == "auto":
        suffix = input_path.suffix.lower()
        if suffix == ".csv":
            input_format = "csv"
        elif suffix == ".json":
            input_format = "json"
        elif suffix == ".jsonl":
            input_format = "jsonl"
        else:
            input_format = "text"
    else:
        input_format = args.input_format

    print(f"Detected input format: {input_format}")

    # Load data
    print("\nLoading data...")
    if input_format == "csv":
        data = load_csv(args.input)
    elif input_format == "json":
        data = load_json(args.input)
    elif input_format == "jsonl":
        data = load_jsonl(args.input)
    else:
        data = load_text(args.input)

    print(f"Loaded {len(data)} examples")

    # Build column mapping
    mapping = {}
    if args.instruction_col:
        mapping["instruction"] = args.instruction_col
    if args.response_col:
        mapping["response"] = args.response_col
    if args.input_col:
        mapping["input"] = args.input_col

    # Convert format
    print("\nConverting format...")
    if args.format == "alpaca":
        converted = convert_to_alpaca(data, mapping)
    elif args.format == "chat":
        converted = convert_to_chat(data)
    else:  # auto
        # Try to detect best format
        converted = convert_to_alpaca(data, mapping)

    print(f"Converted {len(converted)} examples")

    # Split if validation output specified
    if args.val_output:
        train_data, val_data = split_dataset(converted, args.val_ratio, args.seed)
        print(f"\nTrain examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")

        save_jsonl(train_data, args.output)
        save_jsonl(val_data, args.val_output)

        print(f"\nSaved train set to: {args.output}")
        print(f"Saved validation set to: {args.val_output}")
    else:
        save_jsonl(converted, args.output)
        print(f"\nSaved {len(converted)} examples to: {args.output}")

    # Print sample
    print("\n" + "=" * 60)
    print("Sample output:")
    print("=" * 60)
    if converted:
        print(json.dumps(converted[0], indent=2, ensure_ascii=False)[:500])

    print("\n" + "=" * 60)
    print("Done! Use this dataset with:")
    print(f"  python train_qlora.py --data {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
