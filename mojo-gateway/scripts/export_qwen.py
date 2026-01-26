#!/usr/bin/env python3
"""
Export Qwen 2.5 to llama.c binary format for EdgeLLM inference.

Based on karpathy/llama2.c export.py
"""

import os
import sys
import struct
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def export_qwen(model_id: str, output_path: str):
    """Export Qwen model to llama.c binary format."""

    print(f"Loading model: {model_id}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    config = model.config

    # Extract config
    dim = config.hidden_size
    hidden_dim = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    vocab_size = config.vocab_size
    seq_len = getattr(config, 'max_position_embeddings', 2048)

    # Get special token IDs
    eos_token_id = getattr(config, 'eos_token_id', 2)
    bos_token_id = getattr(config, 'bos_token_id', 1)
    if bos_token_id is None:
        bos_token_id = 0  # Use 0 if no BOS (Qwen style)

    print(f"Config:")
    print(f"  dim={dim}, hidden_dim={hidden_dim}")
    print(f"  n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
    print(f"  vocab_size={vocab_size}, seq_len={seq_len}")
    print(f"  bos_token_id={bos_token_id}, eos_token_id={eos_token_id}")

    head_size = dim // n_heads

    # Prepare output file
    print(f"\nExporting to: {output_path}")

    import numpy as np

    kv_dim = n_kv_heads * head_size
    # Limit seq_len for RoPE precomputation (Qwen has 131072 which is too large)
    rope_seq_len = min(seq_len, 8192)

    with open(output_path, 'wb') as f:
        # Write header (7 int32 values)
        # Format: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        # Use limited seq_len for RoPE
        header = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, rope_seq_len)
        f.write(header)

        state_dict = model.state_dict()

        # Helper to write tensor
        def write_tensor(tensor, name=""):
            data = tensor.detach().cpu().float().numpy()
            f.write(data.tobytes())
            print(f"  {name}: {tensor.shape} -> {data.nbytes / 1024 / 1024:.2f} MB")

        # =====================================================================
        # Weight order must match Mojo inference code expectations:
        # All weights for each type are stacked into a single tensor
        # =====================================================================

        # 1. Token embeddings [vocab_size, dim]
        print("\nWriting token_embedding...")
        write_tensor(state_dict['model.embed_tokens.weight'], 'token_embedding')

        # 2. rms_att_weight [n_layers, dim] - stacked into single tensor
        print("\nWriting rms_att_weight [n_layers, dim]...")
        rms_att = torch.stack([state_dict[f'model.layers.{i}.input_layernorm.weight'] for i in range(n_layers)])
        write_tensor(rms_att, 'rms_att_weight')

        # 3. wq [n_layers, dim, dim] - stacked
        print("\nWriting wq [n_layers, dim, dim]...")
        wq = torch.stack([state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] for i in range(n_layers)])
        write_tensor(wq, 'wq')

        # 4. wk [n_layers, kv_dim, dim] - stacked
        print("\nWriting wk [n_layers, kv_dim, dim]...")
        wk = torch.stack([state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] for i in range(n_layers)])
        write_tensor(wk, 'wk')

        # 5. wv [n_layers, kv_dim, dim] - stacked
        print("\nWriting wv [n_layers, kv_dim, dim]...")
        wv = torch.stack([state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] for i in range(n_layers)])
        write_tensor(wv, 'wv')

        # 6. wo [n_layers, dim, dim] - stacked
        print("\nWriting wo [n_layers, dim, dim]...")
        wo = torch.stack([state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] for i in range(n_layers)])
        write_tensor(wo, 'wo')

        # 7. rms_ffn_weight [n_layers, dim] - stacked
        print("\nWriting rms_ffn_weight [n_layers, dim]...")
        rms_ffn = torch.stack([state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] for i in range(n_layers)])
        write_tensor(rms_ffn, 'rms_ffn_weight')

        # 8. w1 (gate_proj) [n_layers, hidden_dim, dim] - stacked
        print("\nWriting w1 [n_layers, hidden_dim, dim]...")
        w1 = torch.stack([state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] for i in range(n_layers)])
        write_tensor(w1, 'w1')

        # 9. w2 (down_proj) [n_layers, dim, hidden_dim] - stacked
        print("\nWriting w2 [n_layers, dim, hidden_dim]...")
        w2 = torch.stack([state_dict[f'model.layers.{i}.mlp.down_proj.weight'] for i in range(n_layers)])
        write_tensor(w2, 'w2')

        # 10. w3 (up_proj) [n_layers, hidden_dim, dim] - stacked
        print("\nWriting w3 [n_layers, hidden_dim, dim]...")
        w3 = torch.stack([state_dict[f'model.layers.{i}.mlp.up_proj.weight'] for i in range(n_layers)])
        write_tensor(w3, 'w3')

        # 11. Final RMS norm [dim]
        print("\nWriting rms_final_weight...")
        write_tensor(state_dict['model.norm.weight'], 'rms_final')

        # 12 & 13. Precompute RoPE frequencies (freq_cis_real, freq_cis_imag)
        # CRITICAL: Use model's rope_theta (Qwen uses 1,000,000, not 10,000!)
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        print(f"\nComputing freq_cis (RoPE with theta={rope_theta})...")
        inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_size, 2, dtype=np.float32) / head_size))
        t = np.arange(rope_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        freq_cis_real = np.cos(freqs).astype(np.float32)
        freq_cis_imag = np.sin(freqs).astype(np.float32)

        f.write(freq_cis_real.tobytes())
        print(f"  freq_cis_real: {freq_cis_real.shape} -> {freq_cis_real.nbytes / 1024 / 1024:.2f} MB")
        f.write(freq_cis_imag.tobytes())
        print(f"  freq_cis_imag: {freq_cis_imag.shape} -> {freq_cis_imag.nbytes / 1024 / 1024:.2f} MB")

        # Note: wcls (output projection) is skipped if weights are shared with embeddings
        # The Mojo code handles this with shared_weights flag

        # 14-16. Q/K/V biases (Qwen has these, most models don't)
        # Check if model has QKV biases
        has_qkv_bias = f'model.layers.0.self_attn.q_proj.bias' in state_dict

        if has_qkv_bias:
            print("\nWriting QKV biases (Qwen-specific)...")

            # bq [n_layers, dim]
            bq = torch.stack([state_dict[f'model.layers.{i}.self_attn.q_proj.bias'] for i in range(n_layers)])
            write_tensor(bq, 'bq')

            # bk [n_layers, kv_dim]
            bk = torch.stack([state_dict[f'model.layers.{i}.self_attn.k_proj.bias'] for i in range(n_layers)])
            write_tensor(bk, 'bk')

            # bv [n_layers, kv_dim]
            bv = torch.stack([state_dict[f'model.layers.{i}.self_attn.v_proj.bias'] for i in range(n_layers)])
            write_tensor(bv, 'bv')
        else:
            print("\nNo QKV biases in this model")

    file_size = os.path.getsize(output_path)
    print(f"\nDone! Output size: {file_size / 1024 / 1024:.2f} MB")
    print(f"Has QKV bias: {has_qkv_bias}")

    # Export tokenizer
    tokenizer_path = output_path.replace('.bin', '_tokenizer.bin')
    print(f"\nExporting tokenizer to: {tokenizer_path}")
    export_tokenizer(tokenizer, tokenizer_path, vocab_size)

    return output_path, tokenizer_path


def export_tokenizer(tokenizer, output_path: str, vocab_size: int):
    """Export tokenizer to llama.c binary format.

    Format (matching Mojo Tokenizer struct expectations):
    - [max_token_length: int32]  (NOT vocab_size first!)
    - For each token (vocab_size times):
        - [score: float32]
        - [token_len: int32]
        - [token_bytes: bytes]
    """

    with open(output_path, 'wb') as f:
        # First pass: collect all tokens and find max length
        max_token_length = 0
        tokens = []
        scores = []

        print(f"  Building tokenizer vocab ({vocab_size} tokens)...")
        for i in range(vocab_size):
            try:
                token = tokenizer.decode([i])
                token_bytes = token.encode('utf-8')
            except:
                token_bytes = b'<unk>'

            tokens.append(token_bytes)
            scores.append(0.0)  # Qwen doesn't use scores like SentencePiece
            max_token_length = max(max_token_length, len(token_bytes))

            if i > 0 and i % 50000 == 0:
                print(f"    Processed {i}/{vocab_size} tokens...")

        # Header: ONLY max_token_length (Mojo expects this, vocab_size comes from model config)
        f.write(struct.pack('i', max_token_length))

        # Write tokens: [score:f32][len:i32][bytes]
        for i, (token_bytes, score) in enumerate(zip(tokens, scores)):
            f.write(struct.pack('f', score))
            f.write(struct.pack('i', len(token_bytes)))
            f.write(token_bytes)

    print(f"Tokenizer exported: {os.path.getsize(output_path) / 1024:.2f} KB (max_token_length={max_token_length})")


def verify_export(model_path: str):
    """Verify the exported model file structure."""
    import numpy as np

    print(f"\n=== Verifying {model_path} ===")

    with open(model_path, 'rb') as f:
        # Read header
        header = struct.unpack('iiiiiii', f.read(28))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = header

        print(f"Header (28 bytes):")
        print(f"  dim={dim}, hidden_dim={hidden_dim}")
        print(f"  n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
        print(f"  vocab_size={vocab_size}, seq_len={seq_len}")

        head_size = dim // n_heads
        kv_dim = n_kv_heads * head_size

        # Expected sizes
        expected = [
            ("token_embedding", vocab_size * dim),
            ("rms_att_weight", n_layers * dim),
            ("wq", n_layers * dim * dim),
            ("wk", n_layers * kv_dim * dim),
            ("wv", n_layers * kv_dim * dim),
            ("wo", n_layers * dim * dim),
            ("rms_ffn_weight", n_layers * dim),
            ("w1", n_layers * hidden_dim * dim),
            ("w2", n_layers * dim * hidden_dim),
            ("w3", n_layers * hidden_dim * dim),
            ("rms_final_weight", dim),
            ("freq_cis_real", seq_len * head_size // 2),
            ("freq_cis_imag", seq_len * head_size // 2),
        ]

        print(f"\nExpected weight structure:")
        total_expected = 28  # header
        for name, size in expected:
            print(f"  {name}: {size:,} floats ({size * 4 / 1024 / 1024:.2f} MB)")
            total_expected += size * 4

        actual_size = os.path.getsize(model_path)
        print(f"\nExpected total: {total_expected / 1024 / 1024:.2f} MB")
        print(f"Actual file size: {actual_size / 1024 / 1024:.2f} MB")

        if abs(actual_size - total_expected) < 100:
            print("✓ File size matches expected!")
        else:
            print(f"✗ Size mismatch: {actual_size - total_expected:,} bytes difference")

        # Read and verify first few weights
        f.seek(28)  # Skip header
        embedding_sample = np.frombuffer(f.read(40), dtype=np.float32)
        print(f"\nFirst 10 embedding values: {embedding_sample[:10]}")

        # Check they're not all zeros or NaN
        if np.all(embedding_sample == 0):
            print("✗ WARNING: Embeddings are all zeros!")
        elif np.any(np.isnan(embedding_sample)):
            print("✗ WARNING: Embeddings contain NaN!")
        else:
            print("✓ Embeddings look valid")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Qwen model to llama.c format')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B',
                        help='HuggingFace model ID')
    parser.add_argument('--output', type=str, default='qwen2.5-1.5b.bin',
                        help='Output binary file')
    parser.add_argument('--verify-only', type=str, default=None,
                        help='Only verify an existing export file')
    args = parser.parse_args()

    if args.verify_only:
        verify_export(args.verify_only)
    else:
        model_path, tokenizer_path = export_qwen(args.model, args.output)
        verify_export(model_path)
