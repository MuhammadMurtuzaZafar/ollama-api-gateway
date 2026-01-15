#!/usr/bin/env python3
"""
Export Qwen 2.5 to INT4 quantized format for EdgeLLM GPU inference.

INT4 quantization with group-wise scaling (AWQ-style):
- Group size: 128 weights per scale
- Symmetric quantization: [-8, 7] mapped to [0, 15]
- Scales stored as float16 for memory efficiency

Memory layout (for efficient GPU upload):
1. Header (32 bytes): 8 int32 values
2. FP32 section: embedding, norms, RoPE, biases
3. INT4 scales section: all FP16 scales together
4. INT4 packed section: all packed weights together

Target: 400+ tok/s on T4 GPU (vs 38 tok/s with FP32)
"""

import os
import struct
import argparse
from typing import Tuple, List
from io import BytesIO

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# INT4 quantization constants
GROUP_SIZE = 128  # Industry standard (AWQ, GPTQ)
QUANT_MIN = -8    # Signed 4-bit range
QUANT_MAX = 7


def quantize_tensor_int4(weights: torch.Tensor) -> Tuple[bytes, bytes]:
    """
    Quantize tensor to INT4 with group-wise scaling.

    Returns:
        scales_bytes: FP16 scales (n_groups * 2 bytes)
        packed_bytes: Packed INT4 values (n_weights / 2 bytes)
    """
    flat = weights.flatten().float()
    n_elements = len(flat)
    n_groups = (n_elements + GROUP_SIZE - 1) // GROUP_SIZE

    # Pad to multiple of GROUP_SIZE
    if n_elements % GROUP_SIZE != 0:
        pad_size = GROUP_SIZE - (n_elements % GROUP_SIZE)
        flat = torch.nn.functional.pad(flat, (0, pad_size), value=0.0)

    scales = []
    packed = bytearray()

    for g in range(n_groups):
        start = g * GROUP_SIZE
        end = start + GROUP_SIZE
        group = flat[start:end]

        # Symmetric quantization: scale = max_abs / 7
        max_abs = torch.max(torch.abs(group)).item()
        scale = max_abs / 7.0 if max_abs > 0 else 1.0
        scales.append(np.float16(scale))

        # Quantize to [-8, 7]
        quant = torch.clamp(torch.round(group / scale), QUANT_MIN, QUANT_MAX)

        # Convert to unsigned [0, 15] for packing
        quant_u = (quant + 8).to(torch.uint8)

        # Pack 2 values per byte (low nibble first)
        for i in range(0, GROUP_SIZE, 2):
            low = quant_u[i].item() & 0x0F
            high = quant_u[i + 1].item() & 0x0F
            packed.append(low | (high << 4))

    scales_array = np.array(scales, dtype=np.float16)
    return scales_array.tobytes(), bytes(packed)


def export_qwen_int4(model_id: str, output_path: str):
    """Export Qwen model to INT4 quantized format."""

    print(f"Loading model: {model_id}")
    print(f"Quantization: INT4 with group size {GROUP_SIZE}")

    # Load model
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

    head_size = dim // n_heads
    kv_dim = n_kv_heads * head_size

    # Limit seq_len for RoPE (Qwen has 131072 which is too large)
    rope_seq_len = min(seq_len, 8192)

    print(f"\nConfig:")
    print(f"  dim={dim}, hidden_dim={hidden_dim}")
    print(f"  n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
    print(f"  vocab_size={vocab_size}, seq_len={rope_seq_len}")

    state_dict = model.state_dict()

    # Check for QKV biases
    has_qkv_bias = f'model.layers.0.self_attn.q_proj.bias' in state_dict

    # Header format:
    # [dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, is_int4]
    is_int4_flag = 1

    print(f"\nExporting to: {output_path}")
    print(f"Memory layout: [Header] [FP32] [INT4 Scales] [INT4 Packed]")

    # Collect all data in separate buffers
    fp32_buffer = BytesIO()
    scales_buffer = BytesIO()
    packed_buffer = BytesIO()

    total_fp32 = 0
    total_scales = 0
    total_packed = 0

    # =========================================================================
    # Section 1: FP32 Data (embedding, norms, RoPE, biases)
    # =========================================================================

    print("\n[Section 1: FP32 Data]")

    # Token embedding
    emb = state_dict['model.embed_tokens.weight'].detach().cpu().float().numpy()
    fp32_buffer.write(emb.tobytes())
    total_fp32 += emb.nbytes
    print(f"  token_embedding: {emb.shape} -> {emb.nbytes / 1024 / 1024:.2f} MB")

    # RMS attention weights
    rms_att = torch.stack([state_dict[f'model.layers.{i}.input_layernorm.weight'] for i in range(n_layers)])
    rms_att_np = rms_att.detach().cpu().float().numpy()
    fp32_buffer.write(rms_att_np.tobytes())
    total_fp32 += rms_att_np.nbytes
    print(f"  rms_att: {rms_att_np.shape} -> {rms_att_np.nbytes / 1024 / 1024:.2f} MB")

    # RMS FFN weights
    rms_ffn = torch.stack([state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] for i in range(n_layers)])
    rms_ffn_np = rms_ffn.detach().cpu().float().numpy()
    fp32_buffer.write(rms_ffn_np.tobytes())
    total_fp32 += rms_ffn_np.nbytes
    print(f"  rms_ffn: {rms_ffn_np.shape} -> {rms_ffn_np.nbytes / 1024 / 1024:.2f} MB")

    # RMS final
    rms_final = state_dict['model.norm.weight'].detach().cpu().float().numpy()
    fp32_buffer.write(rms_final.tobytes())
    total_fp32 += rms_final.nbytes
    print(f"  rms_final: {rms_final.shape} -> {rms_final.nbytes / 1024:.2f} KB")

    # RoPE frequencies
    rope_theta = getattr(config, 'rope_theta', 10000.0)
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_size, 2, dtype=np.float32) / head_size))
    t = np.arange(rope_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    freq_cos = np.cos(freqs).astype(np.float32)
    freq_sin = np.sin(freqs).astype(np.float32)
    fp32_buffer.write(freq_cos.tobytes())
    fp32_buffer.write(freq_sin.tobytes())
    total_fp32 += freq_cos.nbytes + freq_sin.nbytes
    print(f"  freq_cos/sin: {freq_cos.shape} -> {(freq_cos.nbytes + freq_sin.nbytes) / 1024 / 1024:.2f} MB")

    # QKV biases
    if has_qkv_bias:
        bq = torch.stack([state_dict[f'model.layers.{i}.self_attn.q_proj.bias'] for i in range(n_layers)])
        bk = torch.stack([state_dict[f'model.layers.{i}.self_attn.k_proj.bias'] for i in range(n_layers)])
        bv = torch.stack([state_dict[f'model.layers.{i}.self_attn.v_proj.bias'] for i in range(n_layers)])
        bq_np = bq.detach().cpu().float().numpy()
        bk_np = bk.detach().cpu().float().numpy()
        bv_np = bv.detach().cpu().float().numpy()
        fp32_buffer.write(bq_np.tobytes())
        fp32_buffer.write(bk_np.tobytes())
        fp32_buffer.write(bv_np.tobytes())
        total_fp32 += bq_np.nbytes + bk_np.nbytes + bv_np.nbytes
        print(f"  biases (q,k,v): {bq_np.shape} -> {(bq_np.nbytes + bk_np.nbytes + bv_np.nbytes) / 1024 / 1024:.2f} MB")

    # =========================================================================
    # Section 2 & 3: INT4 Scales and Packed Weights
    # =========================================================================

    print("\n[Section 2 & 3: INT4 Quantized Weights]")

    def quantize_and_store(tensor: torch.Tensor, name: str):
        """Quantize tensor and write to scales/packed buffers."""
        nonlocal total_scales, total_packed

        scales_bytes, packed_bytes = quantize_tensor_int4(tensor)
        scales_buffer.write(scales_bytes)
        packed_buffer.write(packed_bytes)

        total_scales += len(scales_bytes)
        total_packed += len(packed_bytes)

        original_bytes = tensor.numel() * 4
        quantized_bytes = len(scales_bytes) + len(packed_bytes)
        compression = original_bytes / quantized_bytes

        print(f"  {name}: {tensor.shape} | "
              f"{original_bytes / 1024 / 1024:.2f} MB -> {quantized_bytes / 1024 / 1024:.2f} MB "
              f"({compression:.1f}x)")

    # Attention weights
    wq = torch.stack([state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] for i in range(n_layers)])
    quantize_and_store(wq, 'wq')

    wk = torch.stack([state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] for i in range(n_layers)])
    quantize_and_store(wk, 'wk')

    wv = torch.stack([state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] for i in range(n_layers)])
    quantize_and_store(wv, 'wv')

    wo = torch.stack([state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] for i in range(n_layers)])
    quantize_and_store(wo, 'wo')

    # FFN weights
    w1 = torch.stack([state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] for i in range(n_layers)])
    quantize_and_store(w1, 'w1 (gate)')

    w2 = torch.stack([state_dict[f'model.layers.{i}.mlp.down_proj.weight'] for i in range(n_layers)])
    quantize_and_store(w2, 'w2 (down)')

    w3 = torch.stack([state_dict[f'model.layers.{i}.mlp.up_proj.weight'] for i in range(n_layers)])
    quantize_and_store(w3, 'w3 (up)')

    # =========================================================================
    # Write to file in correct order
    # =========================================================================

    with open(output_path, 'wb') as f:
        # Header (32 bytes)
        header = struct.pack('iiiiiiii',
            dim, hidden_dim, n_layers, n_heads, n_kv_heads,
            vocab_size, rope_seq_len, is_int4_flag)
        f.write(header)

        # FP32 section
        f.write(fp32_buffer.getvalue())

        # INT4 scales section
        f.write(scales_buffer.getvalue())

        # INT4 packed section
        f.write(packed_buffer.getvalue())

    # Summary
    file_size = os.path.getsize(output_path)

    print(f"\n{'='*60}")
    print(f"INT4 Quantization Complete!")
    print(f"{'='*60}")
    print(f"  Header:       32 bytes")
    print(f"  FP32 data:    {total_fp32 / 1024 / 1024:.2f} MB")
    print(f"  INT4 scales:  {total_scales / 1024 / 1024:.2f} MB")
    print(f"  INT4 packed:  {total_packed / 1024 / 1024:.2f} MB")
    print(f"  ---------------------")
    print(f"  Total file:   {file_size / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")

    # Export tokenizer
    tokenizer_path = output_path.replace('.bin', '_tokenizer.bin')
    print(f"\nExporting tokenizer to: {tokenizer_path}")
    export_tokenizer(tokenizer, tokenizer_path, vocab_size)

    return output_path, tokenizer_path


def export_tokenizer(tokenizer, output_path: str, vocab_size: int):
    """Export tokenizer to llama.c binary format."""

    with open(output_path, 'wb') as f:
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
            scores.append(0.0)
            max_token_length = max(max_token_length, len(token_bytes))

            if i > 0 and i % 50000 == 0:
                print(f"    Processed {i}/{vocab_size} tokens...")

        # Header: max_token_length
        f.write(struct.pack('i', max_token_length))

        # Write tokens: [score:f32][len:i32][bytes]
        for token_bytes, score in zip(tokens, scores):
            f.write(struct.pack('f', score))
            f.write(struct.pack('i', len(token_bytes)))
            f.write(token_bytes)

    print(f"  Tokenizer exported: {os.path.getsize(output_path) / 1024:.2f} KB")


def verify_int4_export(model_path: str):
    """Verify INT4 quantized model structure."""

    print(f"\n=== Verifying INT4 Export: {model_path} ===")

    with open(model_path, 'rb') as f:
        # Read header (8 values)
        header = struct.unpack('iiiiiiii', f.read(32))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, is_int4 = header

        print(f"Header (32 bytes):")
        print(f"  dim={dim}, hidden_dim={hidden_dim}")
        print(f"  n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
        print(f"  vocab_size={vocab_size}, seq_len={seq_len}")
        print(f"  is_int4={is_int4} ({'INT4' if is_int4 else 'FP32'})")

        head_size = dim // n_heads
        kv_dim = n_kv_heads * head_size
        n_groups_dim = (dim + GROUP_SIZE - 1) // GROUP_SIZE
        n_groups_hd = (hidden_dim + GROUP_SIZE - 1) // GROUP_SIZE

        # Calculate expected sizes
        fp32_size = (
            vocab_size * dim +       # embedding
            n_layers * dim +         # rms_att
            n_layers * dim +         # rms_ffn
            dim +                    # rms_final
            seq_len * (head_size//2) * 2 +  # freq_cos + freq_sin
            n_layers * dim +         # bq
            n_layers * kv_dim +      # bk
            n_layers * kv_dim        # bv
        ) * 4

        int4_packed_size = (
            n_layers * dim * dim // 2 +      # wq
            n_layers * kv_dim * dim // 2 +   # wk
            n_layers * kv_dim * dim // 2 +   # wv
            n_layers * dim * dim // 2 +      # wo
            n_layers * hidden_dim * dim // 2 +   # w1
            n_layers * dim * hidden_dim // 2 +   # w2
            n_layers * hidden_dim * dim // 2     # w3
        )

        int4_scales_size = (
            n_layers * dim * n_groups_dim +        # wq
            n_layers * kv_dim * n_groups_dim +     # wk
            n_layers * kv_dim * n_groups_dim +     # wv
            n_layers * dim * n_groups_dim +        # wo
            n_layers * hidden_dim * n_groups_dim + # w1
            n_layers * dim * n_groups_hd +         # w2
            n_layers * hidden_dim * n_groups_dim   # w3
        ) * 2  # FP16

        expected_total = 32 + fp32_size + int4_scales_size + int4_packed_size
        actual_size = os.path.getsize(model_path)

        print(f"\nExpected sizes:")
        print(f"  Header:      32 bytes")
        print(f"  FP32:        {fp32_size / 1024 / 1024:.2f} MB")
        print(f"  INT4 scales: {int4_scales_size / 1024 / 1024:.2f} MB")
        print(f"  INT4 packed: {int4_packed_size / 1024 / 1024:.2f} MB")
        print(f"  Expected:    {expected_total / 1024 / 1024:.2f} MB")
        print(f"  Actual:      {actual_size / 1024 / 1024:.2f} MB")

        # Read first embedding values
        f.seek(32)
        embedding_sample = np.frombuffer(f.read(40), dtype=np.float32)
        print(f"\nFirst 10 embedding values: {embedding_sample[:10]}")

        if np.all(embedding_sample == 0):
            print("WARNING: Embeddings are all zeros!")
        elif np.any(np.isnan(embedding_sample)):
            print("WARNING: Embeddings contain NaN!")
        else:
            print("Embeddings look valid")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Qwen model to INT4 format')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B',
                        help='HuggingFace model ID')
    parser.add_argument('--output', type=str, default='qwen2.5-1.5b_int4.bin',
                        help='Output binary file')
    parser.add_argument('--verify-only', type=str, default=None,
                        help='Only verify an existing export')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Group size for quantization (default: 128)')
    args = parser.parse_args()

    if args.group_size != GROUP_SIZE:
        GROUP_SIZE = args.group_size
        print(f"Using custom group size: {GROUP_SIZE}")

    if args.verify_only:
        verify_int4_export(args.verify_only)
    else:
        model_path, tokenizer_path = export_qwen_int4(args.model, args.output)
        verify_int4_export(model_path)
