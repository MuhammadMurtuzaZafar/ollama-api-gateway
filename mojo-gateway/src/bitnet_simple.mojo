"""
BitNet b1.58 Simple Inference - Flat Weight Storage

Simplified version using flat arrays for all weights.
Based on bitnet.cpp I2_S approach (2-bit weights with scale).
"""
from algorithm import parallelize
from collections import List
from memory import UnsafePointer
from sys import argv
from sys.info import num_performance_cores
import math
import random
import time


# =============================================================================
# Simple T-MAC MatMul (inline, no structs)
# =============================================================================

fn tmac_matmul(
    mut output: List[Float32],
    out_offset: Int,
    activations: List[Float32],
    act_offset: Int,
    weights: List[UInt8],
    scales: List[Float32],
    w_offset: Int,  # Byte offset into weights
    scale_offset: Int,  # Scale offset
    rows: Int,
    cols: Int
):
    """T-MAC matmul with flat weight storage."""
    var w_ptr = weights.unsafe_ptr()
    var s_ptr = scales.unsafe_ptr()
    var a_ptr = activations.unsafe_ptr()
    var bytes_per_row = (cols + 3) // 4

    @parameter
    fn compute_row(row: Int):
        var sum: Float32 = 0.0
        var w_base = w_offset + row * bytes_per_row

        # Process 4 weights at a time
        for col_group in range(0, cols, 4):
            var byte_idx = col_group // 4
            var byte_val = Int(w_ptr[w_base + byte_idx])

            # Decode 4 ternary weights
            for i in range(4):
                if col_group + i >= cols:
                    break
                var bits = (byte_val >> (i * 2)) & 0x03
                var w: Float32
                if bits == 0:
                    w = 0.0
                elif bits == 1:
                    w = 1.0
                else:
                    w = -1.0
                sum += w * a_ptr[act_offset + col_group + i]

        # Apply scale
        output[out_offset + row] = sum * s_ptr[scale_offset + row]

    parallelize[compute_row](rows)


# =============================================================================
# Basic Operations
# =============================================================================

@always_inline
fn rmsnorm(mut output: List[Float32], input: List[Float32], weight: List[Float32],
           o_offset: Int, i_offset: Int, w_offset: Int, size: Int):
    var ss: Float32 = 0.0
    for i in range(size):
        ss += input[i_offset + i] * input[i_offset + i]
    ss = 1.0 / math.sqrt(ss / Float32(size) + 1e-5)
    for i in range(size):
        output[o_offset + i] = weight[w_offset + i] * (ss * input[i_offset + i])


@always_inline
fn softmax(mut x: List[Float32], offset: Int, size: Int):
    var max_val = x[offset]
    for i in range(1, size):
        if x[offset + i] > max_val:
            max_val = x[offset + i]
    var sum_exp: Float32 = 0.0
    for i in range(size):
        x[offset + i] = math.exp(x[offset + i] - max_val)
        sum_exp += x[offset + i]
    for i in range(size):
        x[offset + i] /= sum_exp


@always_inline
fn relu2(mut x: List[Float32], offset: Int, size: Int):
    """ReLU squared activation (BitNet specific)."""
    for i in range(size):
        var val = x[offset + i]
        if val > 0:
            x[offset + i] = val * val
        else:
            x[offset + i] = 0.0


# =============================================================================
# Config and State
# =============================================================================

struct Config:
    var dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var vocab_size: Int
    var seq_len: Int
    var head_size: Int
    var kv_dim: Int
    var kv_mul: Int
    var rope_theta: Float32

    fn __init__(out self, dim: Int, hidden_dim: Int, n_layers: Int,
                n_heads: Int, n_kv_heads: Int, vocab_size: Int, seq_len: Int):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.head_size = dim // n_heads
        self.kv_dim = n_kv_heads * self.head_size
        self.kv_mul = n_heads // n_kv_heads
        self.rope_theta = 500000.0  # BitNet uses higher theta


struct RunState:
    var x: List[Float32]
    var xb: List[Float32]
    var xb2: List[Float32]
    var hb: List[Float32]
    var hb2: List[Float32]
    var q: List[Float32]
    var k: List[Float32]
    var v: List[Float32]
    var att: List[Float32]
    var logits: List[Float32]
    var key_cache: List[Float32]
    var value_cache: List[Float32]

    fn __init__(out self, config: Config):
        self.x = List[Float32]()
        self.xb = List[Float32]()
        self.xb2 = List[Float32]()
        self.hb = List[Float32]()
        self.hb2 = List[Float32]()
        self.q = List[Float32]()
        self.k = List[Float32]()
        self.v = List[Float32]()
        self.att = List[Float32]()
        self.logits = List[Float32]()
        self.key_cache = List[Float32]()
        self.value_cache = List[Float32]()

        for _ in range(config.dim):
            self.x.append(0.0)
            self.xb.append(0.0)
            self.xb2.append(0.0)
            self.q.append(0.0)
        for _ in range(config.kv_dim):
            self.k.append(0.0)
            self.v.append(0.0)
        for _ in range(config.hidden_dim):
            self.hb.append(0.0)
            self.hb2.append(0.0)
        for _ in range(config.n_heads * config.seq_len):
            self.att.append(0.0)
        for _ in range(config.vocab_size):
            self.logits.append(0.0)
        for _ in range(config.n_layers * config.seq_len * config.kv_dim):
            self.key_cache.append(0.0)
            self.value_cache.append(0.0)


# =============================================================================
# Flat Weights Storage with Layer Offsets
# =============================================================================

struct LayerOffsets:
    """Offsets for one layer's weights."""
    var input_norm: Int      # Float offset
    var q_weight: Int        # Ternary byte offset
    var q_scale: Int         # Scale offset
    var k_weight: Int
    var k_scale: Int
    var v_weight: Int
    var v_scale: Int
    var o_weight: Int
    var o_scale: Int
    var attn_sub_norm: Int   # Float offset
    var post_norm: Int       # Float offset
    var gate_weight: Int
    var gate_scale: Int
    var up_weight: Int
    var up_scale: Int
    var down_weight: Int
    var down_scale: Int
    var ffn_sub_norm: Int    # Float offset

    fn __init__(out self):
        self.input_norm = 0
        self.q_weight = 0
        self.q_scale = 0
        self.k_weight = 0
        self.k_scale = 0
        self.v_weight = 0
        self.v_scale = 0
        self.o_weight = 0
        self.o_scale = 0
        self.attn_sub_norm = 0
        self.post_norm = 0
        self.gate_weight = 0
        self.gate_scale = 0
        self.up_weight = 0
        self.up_scale = 0
        self.down_weight = 0
        self.down_scale = 0
        self.ffn_sub_norm = 0


struct FlatWeights(Movable):
    """All weights stored in flat arrays."""
    # Ternary weight data (packed 2-bit)
    var ternary_data: List[UInt8]
    # Per-row scales for ternary weights
    var scales: List[Float32]
    # Float32 weights (norms)
    var float_data: List[Float32]

    # Embedding offsets
    var embed_offset: Int
    var embed_scale_offset: Int

    # LM head offsets
    var lm_head_offset: Int
    var lm_head_scale_offset: Int

    # Final norm
    var final_norm_offset: Int

    # Layer-specific offsets (stored as simple lists for each field)
    var layer_input_norm: List[Int]
    var layer_q_weight: List[Int]
    var layer_q_scale: List[Int]
    var layer_k_weight: List[Int]
    var layer_k_scale: List[Int]
    var layer_v_weight: List[Int]
    var layer_v_scale: List[Int]
    var layer_o_weight: List[Int]
    var layer_o_scale: List[Int]
    var layer_attn_sub_norm: List[Int]
    var layer_post_norm: List[Int]
    var layer_gate_weight: List[Int]
    var layer_gate_scale: List[Int]
    var layer_up_weight: List[Int]
    var layer_up_scale: List[Int]
    var layer_down_weight: List[Int]
    var layer_down_scale: List[Int]
    var layer_ffn_sub_norm: List[Int]

    fn __init__(out self):
        self.ternary_data = List[UInt8]()
        self.scales = List[Float32]()
        self.float_data = List[Float32]()
        self.embed_offset = 0
        self.embed_scale_offset = 0
        self.lm_head_offset = 0
        self.lm_head_scale_offset = 0
        self.final_norm_offset = 0

        self.layer_input_norm = List[Int]()
        self.layer_q_weight = List[Int]()
        self.layer_q_scale = List[Int]()
        self.layer_k_weight = List[Int]()
        self.layer_k_scale = List[Int]()
        self.layer_v_weight = List[Int]()
        self.layer_v_scale = List[Int]()
        self.layer_o_weight = List[Int]()
        self.layer_o_scale = List[Int]()
        self.layer_attn_sub_norm = List[Int]()
        self.layer_post_norm = List[Int]()
        self.layer_gate_weight = List[Int]()
        self.layer_gate_scale = List[Int]()
        self.layer_up_weight = List[Int]()
        self.layer_up_scale = List[Int]()
        self.layer_down_weight = List[Int]()
        self.layer_down_scale = List[Int]()
        self.layer_ffn_sub_norm = List[Int]()

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor for Movable trait."""
        self.ternary_data = other.ternary_data^
        self.scales = other.scales^
        self.float_data = other.float_data^
        self.embed_offset = other.embed_offset
        self.embed_scale_offset = other.embed_scale_offset
        self.lm_head_offset = other.lm_head_offset
        self.lm_head_scale_offset = other.lm_head_scale_offset
        self.final_norm_offset = other.final_norm_offset
        self.layer_input_norm = other.layer_input_norm^
        self.layer_q_weight = other.layer_q_weight^
        self.layer_q_scale = other.layer_q_scale^
        self.layer_k_weight = other.layer_k_weight^
        self.layer_k_scale = other.layer_k_scale^
        self.layer_v_weight = other.layer_v_weight^
        self.layer_v_scale = other.layer_v_scale^
        self.layer_o_weight = other.layer_o_weight^
        self.layer_o_scale = other.layer_o_scale^
        self.layer_attn_sub_norm = other.layer_attn_sub_norm^
        self.layer_post_norm = other.layer_post_norm^
        self.layer_gate_weight = other.layer_gate_weight^
        self.layer_gate_scale = other.layer_gate_scale^
        self.layer_up_weight = other.layer_up_weight^
        self.layer_up_scale = other.layer_up_scale^
        self.layer_down_weight = other.layer_down_weight^
        self.layer_down_scale = other.layer_down_scale^
        self.layer_ffn_sub_norm = other.layer_ffn_sub_norm^


fn load_weights(path: String, config: Config) raises -> FlatWeights:
    """Load weights from T-MAC v2 format into flat storage."""
    var weights = FlatWeights()

    var f = open(path, "r")

    # Read magic
    var magic_bytes = f.read_bytes(4)
    var magic = String("")
    for i in range(3):
        magic += chr(Int(magic_bytes[i]))
    if magic != "TM2":
        raise Error("Invalid model format")

    # Skip config (already loaded)
    _ = f.read_bytes(7 * 4)

    fn read_ternary_matrix(mut file: FileHandle, mut w: FlatWeights) raises -> Tuple[Int, Int, Int, Int]:
        """Read ternary matrix, return (rows, cols, weight_offset, scale_offset)."""
        var weight_offset = len(w.ternary_data)
        var scale_offset = len(w.scales)

        var flag = file.read_bytes(1)
        if Int(flag[0]) != 1:
            raise Error("Expected quantized flag")

        var dims = file.read_bytes(8)
        var dims_ptr = dims.unsafe_ptr().bitcast[Int32]()
        var rows = Int(dims_ptr[0])
        var cols = Int(dims_ptr[1])

        var bytes_per_row = (cols + 3) // 4

        for row in range(rows):
            # Read scale
            var scale_bytes = file.read_bytes(2)
            var scale_f16 = scale_bytes.unsafe_ptr().bitcast[Float16]()[0]
            w.scales.append(Float32(scale_f16))

            # Read ternary data
            var data = file.read_bytes(bytes_per_row)
            for b in range(bytes_per_row):
                w.ternary_data.append(data[b])

        return (rows, cols, weight_offset, scale_offset)

    fn read_float(mut file: FileHandle, mut w: FlatWeights, size: Int) raises -> Int:
        """Read float32 weights, return offset."""
        var offset = len(w.float_data)
        var flag = file.read_bytes(1)
        var data = file.read_bytes(size * 4)
        var ptr = data.unsafe_ptr().bitcast[Float32]()
        for i in range(size):
            w.float_data.append(ptr[i])
        return offset

    # Calculate sizes
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var kv_dim = config.kv_dim
    var n_layers = config.n_layers
    var vocab_size = config.vocab_size

    # Read embedding
    print("Loading embedding...")
    var embed_info = read_ternary_matrix(f, weights)
    weights.embed_offset = embed_info[2]
    weights.embed_scale_offset = embed_info[3]
    print("  Embedding:", embed_info[0], "x", embed_info[1])

    # Process each layer
    for layer in range(n_layers):
        print("Loading layer", layer + 1, "/", n_layers)

        # Input norm
        var input_norm_off = read_float(f, weights, dim)
        weights.layer_input_norm.append(input_norm_off)

        # Q projection
        var q_info = read_ternary_matrix(f, weights)
        weights.layer_q_weight.append(q_info[2])
        weights.layer_q_scale.append(q_info[3])

        # K projection
        var k_info = read_ternary_matrix(f, weights)
        weights.layer_k_weight.append(k_info[2])
        weights.layer_k_scale.append(k_info[3])

        # V projection
        var v_info = read_ternary_matrix(f, weights)
        weights.layer_v_weight.append(v_info[2])
        weights.layer_v_scale.append(v_info[3])

        # O projection
        var o_info = read_ternary_matrix(f, weights)
        weights.layer_o_weight.append(o_info[2])
        weights.layer_o_scale.append(o_info[3])

        # Attention sub-norm
        var attn_sub_norm_off = read_float(f, weights, dim)
        weights.layer_attn_sub_norm.append(attn_sub_norm_off)

        # Post-attention norm
        var post_norm_off = read_float(f, weights, dim)
        weights.layer_post_norm.append(post_norm_off)

        # Gate projection
        var gate_info = read_ternary_matrix(f, weights)
        weights.layer_gate_weight.append(gate_info[2])
        weights.layer_gate_scale.append(gate_info[3])

        # Up projection
        var up_info = read_ternary_matrix(f, weights)
        weights.layer_up_weight.append(up_info[2])
        weights.layer_up_scale.append(up_info[3])

        # Down projection
        var down_info = read_ternary_matrix(f, weights)
        weights.layer_down_weight.append(down_info[2])
        weights.layer_down_scale.append(down_info[3])

        # FFN sub-norm
        var ffn_sub_norm_off = read_float(f, weights, hidden_dim)
        weights.layer_ffn_sub_norm.append(ffn_sub_norm_off)

    # Final norm
    weights.final_norm_offset = read_float(f, weights, dim)

    # LM head
    print("Loading LM head...")
    var lm_head_info = read_ternary_matrix(f, weights)
    weights.lm_head_offset = lm_head_info[2]
    weights.lm_head_scale_offset = lm_head_info[3]

    f.close()

    print()
    print("Loaded", len(weights.ternary_data) // 1024 // 1024, "MB ternary data")
    print("Loaded", len(weights.scales), "scales")
    print("Loaded", len(weights.float_data), "float values")

    return weights^


# =============================================================================
# Transformer Forward Pass
# =============================================================================

fn rope(mut q: List[Float32], mut k: List[Float32], q_offset: Int, k_offset: Int,
        head_size: Int, n_heads: Int, n_kv_heads: Int, pos: Int, theta: Float32):
    """Apply rotary position embeddings."""
    for h in range(n_heads):
        var q_head_offset = q_offset + h * head_size
        for i in range(0, head_size, 2):
            var freq = 1.0 / (theta ** (Float32(i) / Float32(head_size)))
            var val = Float32(pos) * freq
            var cos_val = math.cos(val)
            var sin_val = math.sin(val)

            var q0 = q[q_head_offset + i]
            var q1 = q[q_head_offset + i + 1]
            q[q_head_offset + i] = q0 * cos_val - q1 * sin_val
            q[q_head_offset + i + 1] = q0 * sin_val + q1 * cos_val

    for h in range(n_kv_heads):
        var k_head_offset = k_offset + h * head_size
        for i in range(0, head_size, 2):
            var freq = 1.0 / (theta ** (Float32(i) / Float32(head_size)))
            var val = Float32(pos) * freq
            var cos_val = math.cos(val)
            var sin_val = math.sin(val)

            var k0 = k[k_head_offset + i]
            var k1 = k[k_head_offset + i + 1]
            k[k_head_offset + i] = k0 * cos_val - k1 * sin_val
            k[k_head_offset + i + 1] = k0 * sin_val + k1 * cos_val


fn forward(
    mut state: RunState,
    weights: FlatWeights,
    config: Config,
    token: Int,
    pos: Int
):
    """Run one forward pass through the transformer."""
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var n_heads = config.n_heads
    var n_kv_heads = config.n_kv_heads
    var head_size = config.head_size
    var kv_dim = config.kv_dim
    var kv_mul = config.kv_mul

    # Get token embedding
    var embed_bytes_per_row = (dim + 3) // 4
    var embed_w_offset = weights.embed_offset + token * embed_bytes_per_row
    var embed_s_offset = weights.embed_scale_offset + token

    # Copy embedding to x (reconstruct from ternary)
    var embed_scale = weights.scales[embed_s_offset]
    for i in range(dim):
        var byte_idx = i // 4
        var bit_pos = (i % 4) * 2
        var byte_val = Int(weights.ternary_data[embed_w_offset + byte_idx])
        var bits = (byte_val >> bit_pos) & 0x03
        var w: Float32
        if bits == 0:
            w = 0.0
        elif bits == 1:
            w = 1.0
        else:
            w = -1.0
        state.x[i] = w * embed_scale

    # Process each layer
    for layer in range(config.n_layers):
        # Input normalization
        rmsnorm(state.xb, state.x, weights.float_data, 0, 0,
                weights.layer_input_norm[layer], dim)

        # QKV projections
        tmac_matmul(state.q, 0, state.xb, 0, weights.ternary_data, weights.scales,
                    weights.layer_q_weight[layer], weights.layer_q_scale[layer], dim, dim)
        tmac_matmul(state.k, 0, state.xb, 0, weights.ternary_data, weights.scales,
                    weights.layer_k_weight[layer], weights.layer_k_scale[layer], kv_dim, dim)
        tmac_matmul(state.v, 0, state.xb, 0, weights.ternary_data, weights.scales,
                    weights.layer_v_weight[layer], weights.layer_v_scale[layer], kv_dim, dim)

        # RoPE
        rope(state.q, state.k, 0, 0, head_size, n_heads, n_kv_heads, pos, config.rope_theta)

        # Cache KV
        var cache_offset = layer * config.seq_len * kv_dim + pos * kv_dim
        for i in range(kv_dim):
            state.key_cache[cache_offset + i] = state.k[i]
            state.value_cache[cache_offset + i] = state.v[i]

        # Multi-head attention with GQA
        @parameter
        fn compute_head(h: Int):
            var q_head_offset = h * head_size
            var att_offset = h * config.seq_len
            var kv_head = h // kv_mul

            # Compute attention scores
            for t in range(pos + 1):
                var k_cache_offset = layer * config.seq_len * kv_dim + t * kv_dim + kv_head * head_size
                var score: Float32 = 0.0
                for i in range(head_size):
                    score += state.q[q_head_offset + i] * state.key_cache[k_cache_offset + i]
                state.att[att_offset + t] = score / math.sqrt(Float32(head_size))

            # Softmax
            softmax(state.att, att_offset, pos + 1)

            # Weighted sum of values
            for i in range(head_size):
                var sum_val: Float32 = 0.0
                for t in range(pos + 1):
                    var v_cache_offset = layer * config.seq_len * kv_dim + t * kv_dim + kv_head * head_size
                    sum_val += state.att[att_offset + t] * state.value_cache[v_cache_offset + i]
                state.xb[q_head_offset + i] = sum_val

        parallelize[compute_head](n_heads)

        # Attention sub-norm (BitNet specific)
        rmsnorm(state.xb2, state.xb, weights.float_data, 0, 0,
                weights.layer_attn_sub_norm[layer], dim)

        # Output projection
        tmac_matmul(state.xb, 0, state.xb2, 0, weights.ternary_data, weights.scales,
                    weights.layer_o_weight[layer], weights.layer_o_scale[layer], dim, dim)

        # Residual connection
        for i in range(dim):
            state.x[i] += state.xb[i]

        # Post-attention norm
        rmsnorm(state.xb, state.x, weights.float_data, 0, 0,
                weights.layer_post_norm[layer], dim)

        # FFN: gate and up projections
        tmac_matmul(state.hb, 0, state.xb, 0, weights.ternary_data, weights.scales,
                    weights.layer_gate_weight[layer], weights.layer_gate_scale[layer], hidden_dim, dim)
        tmac_matmul(state.hb2, 0, state.xb, 0, weights.ternary_data, weights.scales,
                    weights.layer_up_weight[layer], weights.layer_up_scale[layer], hidden_dim, dim)

        # ReLU² activation (BitNet specific) and multiply
        for i in range(hidden_dim):
            var gate_val = state.hb[i]
            if gate_val > 0:
                gate_val = gate_val * gate_val  # ReLU²
            else:
                gate_val = 0.0
            state.hb[i] = gate_val * state.hb2[i]

        # FFN sub-norm (BitNet specific)
        rmsnorm(state.hb2, state.hb, weights.float_data, 0, 0,
                weights.layer_ffn_sub_norm[layer], hidden_dim)

        # Down projection
        tmac_matmul(state.xb, 0, state.hb2, 0, weights.ternary_data, weights.scales,
                    weights.layer_down_weight[layer], weights.layer_down_scale[layer], dim, hidden_dim)

        # Residual connection
        for i in range(dim):
            state.x[i] += state.xb[i]

    # Final norm (use xb as temp buffer to avoid aliasing)
    rmsnorm(state.xb, state.x, weights.float_data, 0, 0,
            weights.final_norm_offset, dim)
    for i in range(dim):
        state.x[i] = state.xb[i]

    # LM head (classifier)
    tmac_matmul(state.logits, 0, state.x, 0, weights.ternary_data, weights.scales,
                weights.lm_head_offset, weights.lm_head_scale_offset, config.vocab_size, dim)


# =============================================================================
# Sampling
# =============================================================================

fn sample_argmax(logits: List[Float32], size: Int) -> Int:
    var max_idx = 0
    var max_val = logits[0]
    for i in range(1, size):
        if logits[i] > max_val:
            max_val = logits[i]
            max_idx = i
    return max_idx


fn sample_topp(mut logits: List[Float32], size: Int, topp: Float32, temp: Float32) -> Int:
    """Top-p (nucleus) sampling with temperature."""
    if temp == 0.0:
        return sample_argmax(logits, size)

    # Apply temperature
    for i in range(size):
        logits[i] /= temp

    # Softmax
    softmax(logits, 0, size)

    # Create list of (probability, index) and sort by probability descending
    # Use a simple approach: find top-p tokens by repeatedly finding max
    var probs = List[Float32]()
    var indices = List[Int]()
    var used = List[Bool]()

    for i in range(size):
        probs.append(logits[i])
        indices.append(i)
        used.append(False)

    # Find tokens in descending probability order until cumprob >= topp
    var cumprob: Float32 = 0.0
    var top_indices = List[Int]()
    var top_probs = List[Float32]()

    while cumprob < topp:
        var max_prob: Float32 = -1.0
        var max_idx = -1
        for i in range(size):
            if not used[i] and probs[i] > max_prob:
                max_prob = probs[i]
                max_idx = i
        if max_idx < 0:
            break
        used[max_idx] = True
        top_indices.append(max_idx)
        top_probs.append(max_prob)
        cumprob += max_prob

    # Renormalize the top-p tokens
    var sum_prob: Float32 = 0.0
    for i in range(len(top_probs)):
        sum_prob += top_probs[i]

    if sum_prob <= 0:
        # Fallback to argmax if something went wrong
        return sample_argmax(logits, size)

    # Sample from top-p distribution
    var r = random.random_float64().cast[DType.float32]() * sum_prob
    cumprob = 0.0
    for i in range(len(top_probs)):
        cumprob += top_probs[i]
        if cumprob >= r:
            return top_indices[i]

    # Fallback: return the highest probability token
    return top_indices[0] if len(top_indices) > 0 else 0


# =============================================================================
# Main
# =============================================================================

fn main() raises:
    var args = argv()
    if len(args) < 2:
        print("Usage: bitnet_simple <model.tmac2.bin> [-n tokens] [-t temp] [-p topp]")
        return

    var model_path = String(args[1])
    var num_tokens = 32
    var temperature: Float32 = 0.8
    var topp: Float32 = 0.9

    var i = 2
    while i < len(args):
        if String(args[i]) == "-n" and i + 1 < len(args):
            num_tokens = atol(args[i + 1])
            i += 2
        elif String(args[i]) == "-t" and i + 1 < len(args):
            temperature = Float32(atof(args[i + 1]))
            i += 2
        elif String(args[i]) == "-p" and i + 1 < len(args):
            topp = Float32(atof(args[i + 1]))
            i += 2
        else:
            i += 1

    print("BitNet b1.58 T-MAC Inference")
    print("=" * 50)
    print("Loading model from", model_path)

    # BitNet 2B config
    var config = Config(
        dim=2560,
        hidden_dim=6912,
        n_layers=30,
        n_heads=20,
        n_kv_heads=5,
        vocab_size=128256,
        seq_len=4096
    )

    print("Config: dim=", config.dim, "hidden=", config.hidden_dim, "layers=", config.n_layers)
    print("        heads=", config.n_heads, "kv_heads=", config.n_kv_heads)

    var weights = load_weights(model_path, config)
    var state = RunState(config)

    print()
    print("Model loaded successfully!")
    print("Ternary data:", len(weights.ternary_data) // 1024 // 1024, "MB")
    print()
    print("Generating", num_tokens, "tokens...")
    print("Temperature:", temperature, "Top-p:", topp)
    print("-" * 50)

    # Start with BOS token (128000 for BitNet/Llama3)
    var token = 128000
    var start_time = time.perf_counter_ns()
    var tokens_generated = 0

    for pos in range(num_tokens):
        # Forward pass
        forward(state, weights, config, token, pos)

        # Sample next token
        if temperature == 0.0:
            token = sample_argmax(state.logits, config.vocab_size)
        else:
            token = sample_topp(state.logits, config.vocab_size, topp, temperature)

        # Print token (as number for now - need tokenizer for text)
        print("Token", pos, ":", token)
        tokens_generated += 1

        # Check for EOS
        if token == 128001:
            print("<EOS>")
            break

    var end_time = time.perf_counter_ns()
    var elapsed_s = Float64(end_time - start_time) / 1e9
    var tok_per_sec = Float64(tokens_generated) / elapsed_s

    print("-" * 50)
    print()
    print("Generated", tokens_generated, "tokens in", elapsed_s, "seconds")
    print("Speed:", tok_per_sec, "tok/s")
    print()
    print("Note: Output shows token IDs. Use a tokenizer to decode to text.")
