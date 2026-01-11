"""
BitNet b1.58 T-MAC LUT Inference - Optimized with Lookup Tables

Key optimizations over bitnet_simple.mojo:
1. Precomputed lookup tables for activation groups (no multiplications in matmul)
2. Per-row scales applied after LUT accumulation
3. Parallel row computation

Based on T-MAC paper: https://arxiv.org/abs/2407.00088
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
# Lookup Table for T-MAC
# =============================================================================

struct LookupTable:
    """
    Precomputed partial sums for activation groups.
    For 4 ternary weights (2 bits each = 1 byte), we have 256 possible patterns.
    LUT[group][pattern] = sum of weight[i] * activation[group*4 + i]
    """
    var tables: List[Float32]
    var num_groups: Int

    fn __init__(out self, num_groups: Int):
        self.num_groups = num_groups
        var total_entries = num_groups * 256
        self.tables = List[Float32](capacity=total_entries)
        for _ in range(total_entries):
            self.tables.append(0.0)

    fn __moveinit__(out self, deinit other: Self):
        self.tables = other.tables^
        self.num_groups = other.num_groups

    @always_inline
    fn get(self, group: Int, pattern: Int) -> Float32:
        return self.tables[group * 256 + pattern]

    @always_inline
    fn set(mut self, group: Int, pattern: Int, value: Float32):
        self.tables[group * 256 + pattern] = value


@always_inline
fn decode_ternary(bits: Int) -> Int:
    """Decode 2-bit pattern to ternary: 00=0, 01=+1, 11=-1, 10=-1."""
    if bits == 0:
        return 0
    elif bits == 1:
        return 1
    else:
        return -1


fn build_lut(activations: List[Float32], offset: Int, size: Int) -> LookupTable:
    """
    Build lookup table for activation vector.
    For each group of 4 activations, precompute all 256 possible weighted sums.
    """
    var num_groups = (size + 3) // 4
    var lut = LookupTable(num_groups)
    var act_ptr = activations.unsafe_ptr()

    for g in range(num_groups):
        var base = offset + g * 4
        # Get 4 activations (with bounds checking)
        var a0 = act_ptr[base] if base < offset + size else Float32(0)
        var a1 = act_ptr[base + 1] if base + 1 < offset + size else Float32(0)
        var a2 = act_ptr[base + 2] if base + 2 < offset + size else Float32(0)
        var a3 = act_ptr[base + 3] if base + 3 < offset + size else Float32(0)

        # Precompute all 256 possible sums for this group
        for pattern in range(256):
            var w0 = decode_ternary((pattern >> 0) & 0x03)
            var w1 = decode_ternary((pattern >> 2) & 0x03)
            var w2 = decode_ternary((pattern >> 4) & 0x03)
            var w3 = decode_ternary((pattern >> 6) & 0x03)

            var sum: Float32 = 0.0
            if w0 == 1:
                sum += a0
            elif w0 == -1:
                sum -= a0
            if w1 == 1:
                sum += a1
            elif w1 == -1:
                sum -= a1
            if w2 == 1:
                sum += a2
            elif w2 == -1:
                sum -= a2
            if w3 == 1:
                sum += a3
            elif w3 == -1:
                sum -= a3

            lut.set(g, pattern, sum)

    return lut^


# =============================================================================
# T-MAC MatMul with LUT
# =============================================================================

fn tmac_matmul_lut(
    mut output: List[Float32],
    out_offset: Int,
    lut: LookupTable,
    weights: List[UInt8],
    scales: List[Float32],
    w_offset: Int,
    scale_offset: Int,
    rows: Int,
    cols: Int
):
    """
    T-MAC matmul using precomputed LUT.
    output[row] = scale[row] * sum_g(LUT[g][weight_pattern[row][g]])

    No multiplications during matmul - just table lookups!
    """
    var w_ptr = weights.unsafe_ptr()
    var s_ptr = scales.unsafe_ptr()
    var bytes_per_row = (cols + 3) // 4
    var num_groups = bytes_per_row

    @parameter
    fn compute_row(row: Int):
        var sum: Float32 = 0.0
        var w_base = w_offset + row * bytes_per_row

        # Pure table lookups - no multiplications
        for g in range(num_groups):
            var pattern = Int(w_ptr[w_base + g])
            sum += lut.get(g, pattern)

        # Apply per-row scale (single multiplication per row)
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
        self.rope_theta = 500000.0


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
# Flat Weights Storage
# =============================================================================

struct FlatWeights(Movable):
    var ternary_data: List[UInt8]
    var scales: List[Float32]
    var float_data: List[Float32]

    var embed_offset: Int
    var embed_scale_offset: Int
    var lm_head_offset: Int
    var lm_head_scale_offset: Int
    var final_norm_offset: Int

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


fn load_config(path: String) raises -> Config:
    """Load config from T-MAC v2 file header."""
    var f = open(path, "r")

    var magic_bytes = f.read_bytes(4)
    var magic = String("")
    for i in range(3):
        magic += chr(Int(magic_bytes[i]))
    if magic != "TM2":
        raise Error("Invalid model format: expected TM2")

    # Read header: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
    var header = f.read_bytes(7 * 4)
    var header_ptr = header.unsafe_ptr().bitcast[Int32]()

    return Config(
        dim=Int(header_ptr[0]),
        hidden_dim=Int(header_ptr[1]),
        n_layers=Int(header_ptr[2]),
        n_heads=Int(header_ptr[3]),
        n_kv_heads=Int(header_ptr[4]),
        vocab_size=Int(header_ptr[5]),
        seq_len=Int(header_ptr[6])
    )


fn load_weights(path: String, config: Config) raises -> FlatWeights:
    """Load weights from T-MAC v2 format."""
    var weights = FlatWeights()

    var f = open(path, "r")

    var magic_bytes = f.read_bytes(4)
    var magic = String("")
    for i in range(3):
        magic += chr(Int(magic_bytes[i]))
    if magic != "TM2":
        raise Error("Invalid model format")

    # Skip header (already read by load_config)
    _ = f.read_bytes(7 * 4)

    fn read_ternary_matrix(mut file: FileHandle, mut w: FlatWeights) raises -> Tuple[Int, Int, Int, Int]:
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
            var scale_bytes = file.read_bytes(2)
            var scale_f16 = scale_bytes.unsafe_ptr().bitcast[Float16]()[0]
            w.scales.append(Float32(scale_f16))

            var data = file.read_bytes(bytes_per_row)
            for b in range(bytes_per_row):
                w.ternary_data.append(data[b])

        return (rows, cols, weight_offset, scale_offset)

    fn read_float(mut file: FileHandle, mut w: FlatWeights, size: Int) raises -> Int:
        var offset = len(w.float_data)
        var flag = file.read_bytes(1)
        var data = file.read_bytes(size * 4)
        var ptr = data.unsafe_ptr().bitcast[Float32]()
        for i in range(size):
            w.float_data.append(ptr[i])
        return offset

    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var n_layers = config.n_layers

    print("Loading embedding...")
    var embed_info = read_ternary_matrix(f, weights)
    weights.embed_offset = embed_info[2]
    weights.embed_scale_offset = embed_info[3]
    print("  Embedding:", embed_info[0], "x", embed_info[1])

    for layer in range(n_layers):
        print("Loading layer", layer + 1, "/", n_layers)

        var input_norm_off = read_float(f, weights, dim)
        weights.layer_input_norm.append(input_norm_off)

        var q_info = read_ternary_matrix(f, weights)
        weights.layer_q_weight.append(q_info[2])
        weights.layer_q_scale.append(q_info[3])

        var k_info = read_ternary_matrix(f, weights)
        weights.layer_k_weight.append(k_info[2])
        weights.layer_k_scale.append(k_info[3])

        var v_info = read_ternary_matrix(f, weights)
        weights.layer_v_weight.append(v_info[2])
        weights.layer_v_scale.append(v_info[3])

        var o_info = read_ternary_matrix(f, weights)
        weights.layer_o_weight.append(o_info[2])
        weights.layer_o_scale.append(o_info[3])

        var attn_sub_norm_off = read_float(f, weights, dim)
        weights.layer_attn_sub_norm.append(attn_sub_norm_off)

        var post_norm_off = read_float(f, weights, dim)
        weights.layer_post_norm.append(post_norm_off)

        var gate_info = read_ternary_matrix(f, weights)
        weights.layer_gate_weight.append(gate_info[2])
        weights.layer_gate_scale.append(gate_info[3])

        var up_info = read_ternary_matrix(f, weights)
        weights.layer_up_weight.append(up_info[2])
        weights.layer_up_scale.append(up_info[3])

        var down_info = read_ternary_matrix(f, weights)
        weights.layer_down_weight.append(down_info[2])
        weights.layer_down_scale.append(down_info[3])

        var ffn_sub_norm_off = read_float(f, weights, hidden_dim)
        weights.layer_ffn_sub_norm.append(ffn_sub_norm_off)

    weights.final_norm_offset = read_float(f, weights, dim)

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
# Transformer Forward Pass with T-MAC LUT
# =============================================================================

fn rope(mut q: List[Float32], mut k: List[Float32], q_offset: Int, k_offset: Int,
        head_size: Int, n_heads: Int, n_kv_heads: Int, pos: Int, theta: Float32):
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
    """Forward pass using T-MAC LUT optimization."""
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

        # Build LUT for normalized input (dim-sized vector)
        var lut_dim = build_lut(state.xb, 0, dim)

        # QKV projections using LUT
        tmac_matmul_lut(state.q, 0, lut_dim, weights.ternary_data, weights.scales,
                        weights.layer_q_weight[layer], weights.layer_q_scale[layer], dim, dim)
        tmac_matmul_lut(state.k, 0, lut_dim, weights.ternary_data, weights.scales,
                        weights.layer_k_weight[layer], weights.layer_k_scale[layer], kv_dim, dim)
        tmac_matmul_lut(state.v, 0, lut_dim, weights.ternary_data, weights.scales,
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

            for t in range(pos + 1):
                var k_cache_offset = layer * config.seq_len * kv_dim + t * kv_dim + kv_head * head_size
                var score: Float32 = 0.0
                for i in range(head_size):
                    score += state.q[q_head_offset + i] * state.key_cache[k_cache_offset + i]
                state.att[att_offset + t] = score / math.sqrt(Float32(head_size))

            softmax(state.att, att_offset, pos + 1)

            for i in range(head_size):
                var sum_val: Float32 = 0.0
                for t in range(pos + 1):
                    var v_cache_offset = layer * config.seq_len * kv_dim + t * kv_dim + kv_head * head_size
                    sum_val += state.att[att_offset + t] * state.value_cache[v_cache_offset + i]
                state.xb[q_head_offset + i] = sum_val

        parallelize[compute_head](n_heads)

        # Attention sub-norm
        rmsnorm(state.xb2, state.xb, weights.float_data, 0, 0,
                weights.layer_attn_sub_norm[layer], dim)

        # Build LUT for attention output
        var lut_attn = build_lut(state.xb2, 0, dim)

        # Output projection using LUT
        tmac_matmul_lut(state.xb, 0, lut_attn, weights.ternary_data, weights.scales,
                        weights.layer_o_weight[layer], weights.layer_o_scale[layer], dim, dim)

        # Residual connection
        for i in range(dim):
            state.x[i] += state.xb[i]

        # Post-attention norm
        rmsnorm(state.xb, state.x, weights.float_data, 0, 0,
                weights.layer_post_norm[layer], dim)

        # Build LUT for FFN input
        var lut_ffn = build_lut(state.xb, 0, dim)

        # FFN: gate and up projections using LUT
        tmac_matmul_lut(state.hb, 0, lut_ffn, weights.ternary_data, weights.scales,
                        weights.layer_gate_weight[layer], weights.layer_gate_scale[layer], hidden_dim, dim)
        tmac_matmul_lut(state.hb2, 0, lut_ffn, weights.ternary_data, weights.scales,
                        weights.layer_up_weight[layer], weights.layer_up_scale[layer], hidden_dim, dim)

        # ReLU squared and multiply
        for i in range(hidden_dim):
            var gate_val = state.hb[i]
            if gate_val > 0:
                gate_val = gate_val * gate_val
            else:
                gate_val = 0.0
            state.hb[i] = gate_val * state.hb2[i]

        # FFN sub-norm
        rmsnorm(state.hb2, state.hb, weights.float_data, 0, 0,
                weights.layer_ffn_sub_norm[layer], hidden_dim)

        # Build LUT for FFN down projection (hidden_dim sized)
        var lut_down = build_lut(state.hb2, 0, hidden_dim)

        # Down projection using LUT
        tmac_matmul_lut(state.xb, 0, lut_down, weights.ternary_data, weights.scales,
                        weights.layer_down_weight[layer], weights.layer_down_scale[layer], dim, hidden_dim)

        # Residual connection
        for i in range(dim):
            state.x[i] += state.xb[i]

    # Final norm
    rmsnorm(state.xb, state.x, weights.float_data, 0, 0,
            weights.final_norm_offset, dim)
    for i in range(dim):
        state.x[i] = state.xb[i]

    # Build LUT for LM head
    var lut_lm = build_lut(state.x, 0, dim)

    # LM head using LUT
    tmac_matmul_lut(state.logits, 0, lut_lm, weights.ternary_data, weights.scales,
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
    """Top-p sampling with temperature."""
    if temp == 0.0:
        return sample_argmax(logits, size)

    for i in range(size):
        logits[i] /= temp

    softmax(logits, 0, size)

    var probs = List[Float32]()
    var used = List[Bool]()
    for i in range(size):
        probs.append(logits[i])
        used.append(False)

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

    var sum_prob: Float32 = 0.0
    for i in range(len(top_probs)):
        sum_prob += top_probs[i]

    if sum_prob <= 0:
        return sample_argmax(logits, size)

    var r = random.random_float64().cast[DType.float32]() * sum_prob
    cumprob = 0.0
    for i in range(len(top_probs)):
        cumprob += top_probs[i]
        if cumprob >= r:
            return top_indices[i]

    return top_indices[0] if len(top_indices) > 0 else 0


# =============================================================================
# Main
# =============================================================================

fn main() raises:
    var args = argv()
    if len(args) < 2:
        print("Usage: bitnet_tmac_lut <model.tmac2.bin> [-n tokens] [-t temp] [-p topp]")
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

    print("BitNet b1.58 T-MAC LUT Inference")
    print("=" * 50)
    print("Loading model from", model_path)

    # Load config from file header
    var config = load_config(model_path)

    print("Config: dim=", config.dim, "hidden=", config.hidden_dim, "layers=", config.n_layers)
    print("        heads=", config.n_heads, "kv_heads=", config.n_kv_heads)
    print("        vocab_size=", config.vocab_size, "seq_len=", config.seq_len)

    var weights = load_weights(model_path, config)
    var state = RunState(config)

    print()
    print("Model loaded successfully!")
    print("Ternary data:", len(weights.ternary_data) // 1024 // 1024, "MB")
    print()
    print("Generating", num_tokens, "tokens...")
    print("Temperature:", temperature, "Top-p:", topp)
    print("-" * 50)

    # Use BOS token (typically 0 or 1, clamped to vocab size for safety)
    var token = 1 if config.vocab_size > 1 else 0
    var start_time = time.perf_counter_ns()
    var tokens_generated = 0

    for pos in range(num_tokens):
        forward(state, weights, config, token, pos)

        if temperature == 0.0:
            token = sample_argmax(state.logits, config.vocab_size)
        else:
            token = sample_topp(state.logits, config.vocab_size, topp, temperature)

        print("Token", pos, ":", token)
        tokens_generated += 1

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
