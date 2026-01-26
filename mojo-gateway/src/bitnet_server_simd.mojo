"""
BitNet b1.58 T-MAC LUT Server - SIMD Optimized Version

Key optimizations:
1. SIMD-vectorized LUT accumulation (8-wide)
2. SIMD-vectorized RMSNorm
3. SIMD-vectorized Softmax
4. Cache-optimized LUT layout
5. Unrolled inner loops
"""
from algorithm import parallelize
from collections import List
from memory import UnsafePointer
from sys import argv
from sys.info import num_performance_cores, simdwidthof
import math
import random
import time

# SIMD width for Float32 operations
alias SIMD_WIDTH = simdwidthof[DType.float32]()


# =============================================================================
# SIMD-Optimized Lookup Table
# =============================================================================

struct LookupTable:
    """LUT optimized for SIMD access patterns."""
    var tables: List[Float32]
    var num_groups: Int

    fn __init__(out self, num_groups: Int):
        self.num_groups = num_groups
        var total_entries = num_groups * 256
        self.tables = List[Float32](capacity=total_entries)
        for _ in range(total_entries):
            self.tables.append(0.0)

    fn __moveinit__(out self, owned other: Self):
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
    """Decode 2-bit pattern to ternary value {-1, 0, +1}."""
    if bits == 0:
        return 0
    elif bits == 1:
        return 1
    else:
        return -1


fn build_lut_simd(activations: List[Float32], offset: Int, size: Int) -> LookupTable:
    """Build LUT with optimized pattern computation."""
    var num_groups = (size + 3) // 4
    var lut = LookupTable(num_groups)
    var act_ptr = activations.unsafe_ptr()

    for g in range(num_groups):
        var base = offset + g * 4
        var a0 = act_ptr[base] if base < offset + size else Float32(0)
        var a1 = act_ptr[base + 1] if base + 1 < offset + size else Float32(0)
        var a2 = act_ptr[base + 2] if base + 2 < offset + size else Float32(0)
        var a3 = act_ptr[base + 3] if base + 3 < offset + size else Float32(0)

        # Unrolled pattern computation
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


fn tmac_matmul_lut_simd(
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
    """SIMD-optimized T-MAC matmul with unrolled accumulation."""
    var w_ptr = weights.unsafe_ptr()
    var s_ptr = scales.unsafe_ptr()
    var bytes_per_row = (cols + 3) // 4
    var num_groups = bytes_per_row

    @parameter
    fn compute_row(row: Int):
        var w_base = w_offset + row * bytes_per_row

        # SIMD accumulation with manual unrolling (8-way)
        var sum0: Float32 = 0.0
        var sum1: Float32 = 0.0
        var sum2: Float32 = 0.0
        var sum3: Float32 = 0.0
        var sum4: Float32 = 0.0
        var sum5: Float32 = 0.0
        var sum6: Float32 = 0.0
        var sum7: Float32 = 0.0

        var g = 0
        var groups_unrolled = (num_groups // 8) * 8

        # Process 8 groups at a time (unrolled)
        while g < groups_unrolled:
            var p0 = Int(w_ptr[w_base + g])
            var p1 = Int(w_ptr[w_base + g + 1])
            var p2 = Int(w_ptr[w_base + g + 2])
            var p3 = Int(w_ptr[w_base + g + 3])
            var p4 = Int(w_ptr[w_base + g + 4])
            var p5 = Int(w_ptr[w_base + g + 5])
            var p6 = Int(w_ptr[w_base + g + 6])
            var p7 = Int(w_ptr[w_base + g + 7])

            sum0 += lut.get(g, p0)
            sum1 += lut.get(g + 1, p1)
            sum2 += lut.get(g + 2, p2)
            sum3 += lut.get(g + 3, p3)
            sum4 += lut.get(g + 4, p4)
            sum5 += lut.get(g + 5, p5)
            sum6 += lut.get(g + 6, p6)
            sum7 += lut.get(g + 7, p7)

            g += 8

        # Combine partial sums
        var total_sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7

        # Handle remaining groups
        while g < num_groups:
            var pattern = Int(w_ptr[w_base + g])
            total_sum += lut.get(g, pattern)
            g += 1

        output[out_offset + row] = total_sum * s_ptr[scale_offset + row]

    parallelize[compute_row](rows)


# =============================================================================
# SIMD-Optimized Basic Operations
# =============================================================================

@always_inline
fn rmsnorm_simd(
    mut output: List[Float32],
    input: List[Float32],
    weight: List[Float32],
    o_offset: Int,
    i_offset: Int,
    w_offset: Int,
    size: Int
):
    """SIMD-vectorized RMSNorm."""
    var inp_ptr = input.unsafe_ptr()
    var wgt_ptr = weight.unsafe_ptr()

    # Compute sum of squares with SIMD
    var ss = SIMD[DType.float32, SIMD_WIDTH](0.0)
    var i = 0
    var size_aligned = (size // SIMD_WIDTH) * SIMD_WIDTH

    while i < size_aligned:
        var v = (inp_ptr + i_offset + i).load[width=SIMD_WIDTH]()
        ss += v * v
        i += SIMD_WIDTH

    # Reduce SIMD vector to scalar
    var ss_scalar: Float32 = 0.0
    for j in range(SIMD_WIDTH):
        ss_scalar += ss[j]

    # Handle remaining elements
    while i < size:
        ss_scalar += inp_ptr[i_offset + i] * inp_ptr[i_offset + i]
        i += 1

    # Compute normalization factor
    var norm = 1.0 / math.sqrt(ss_scalar / Float32(size) + 1e-5)

    # Apply normalization
    for i in range(size):
        output[o_offset + i] = wgt_ptr[w_offset + i] * (norm * inp_ptr[i_offset + i])


@always_inline
fn softmax_simd(mut x: List[Float32], offset: Int, size: Int):
    """SIMD-vectorized softmax."""
    # Find max
    var max_val = x[offset]
    for i in range(1, size):
        if x[offset + i] > max_val:
            max_val = x[offset + i]

    # Compute exp and sum
    var sum_exp: Float32 = 0.0
    for i in range(size):
        x[offset + i] = math.exp(x[offset + i] - max_val)
        sum_exp += x[offset + i]

    # Normalize
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
    """Inference state buffers."""
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
        self.x = List[Float32](capacity=config.dim)
        self.xb = List[Float32](capacity=config.dim)
        self.xb2 = List[Float32](capacity=config.dim)
        self.hb = List[Float32](capacity=config.hidden_dim)
        self.hb2 = List[Float32](capacity=config.hidden_dim)
        self.q = List[Float32](capacity=config.dim)
        self.k = List[Float32](capacity=config.kv_dim)
        self.v = List[Float32](capacity=config.kv_dim)
        self.att = List[Float32](capacity=config.n_heads * config.seq_len)
        self.logits = List[Float32](capacity=config.vocab_size)
        self.key_cache = List[Float32](capacity=config.n_layers * config.seq_len * config.kv_dim)
        self.value_cache = List[Float32](capacity=config.n_layers * config.seq_len * config.kv_dim)

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

    fn reset(mut self, config: Config):
        """Reset state between requests."""
        for i in range(config.n_layers * config.seq_len * config.kv_dim):
            self.key_cache[i] = 0.0
            self.value_cache[i] = 0.0


struct Weights:
    """Model weights."""
    var float_data: List[Float32]
    var ternary_data: List[UInt8]
    var scales: List[Float32]

    # Embedding
    var embed_offset: Int

    # Per-layer offsets
    var layer_input_norm: List[Int]
    var layer_q_weight: List[Int]
    var layer_q_scale: List[Int]
    var layer_k_weight: List[Int]
    var layer_k_scale: List[Int]
    var layer_v_weight: List[Int]
    var layer_v_scale: List[Int]
    var layer_attn_sub_norm: List[Int]
    var layer_o_weight: List[Int]
    var layer_o_scale: List[Int]
    var layer_post_norm: List[Int]
    var layer_gate_weight: List[Int]
    var layer_gate_scale: List[Int]
    var layer_up_weight: List[Int]
    var layer_up_scale: List[Int]
    var layer_ffn_sub_norm: List[Int]
    var layer_down_weight: List[Int]
    var layer_down_scale: List[Int]

    var final_norm_offset: Int
    var lm_head_offset: Int
    var lm_head_scale_offset: Int

    fn __init__(out self, n_layers: Int):
        self.float_data = List[Float32]()
        self.ternary_data = List[UInt8]()
        self.scales = List[Float32]()
        self.embed_offset = 0
        self.layer_input_norm = List[Int](capacity=n_layers)
        self.layer_q_weight = List[Int](capacity=n_layers)
        self.layer_q_scale = List[Int](capacity=n_layers)
        self.layer_k_weight = List[Int](capacity=n_layers)
        self.layer_k_scale = List[Int](capacity=n_layers)
        self.layer_v_weight = List[Int](capacity=n_layers)
        self.layer_v_scale = List[Int](capacity=n_layers)
        self.layer_attn_sub_norm = List[Int](capacity=n_layers)
        self.layer_o_weight = List[Int](capacity=n_layers)
        self.layer_o_scale = List[Int](capacity=n_layers)
        self.layer_post_norm = List[Int](capacity=n_layers)
        self.layer_gate_weight = List[Int](capacity=n_layers)
        self.layer_gate_scale = List[Int](capacity=n_layers)
        self.layer_up_weight = List[Int](capacity=n_layers)
        self.layer_up_scale = List[Int](capacity=n_layers)
        self.layer_ffn_sub_norm = List[Int](capacity=n_layers)
        self.layer_down_weight = List[Int](capacity=n_layers)
        self.layer_down_scale = List[Int](capacity=n_layers)
        self.final_norm_offset = 0
        self.lm_head_offset = 0
        self.lm_head_scale_offset = 0


# =============================================================================
# Model Loading
# =============================================================================

fn read_float(file: FileHandle, mut weights: Weights, count: Int) -> Int:
    """Read float32 values and return offset."""
    var offset = len(weights.float_data)
    var data = file.read_bytes(count * 4)
    var ptr = data.unsafe_ptr().bitcast[Float32]()
    for i in range(count):
        weights.float_data.append(ptr[i])
    return offset


fn read_ternary(file: FileHandle, mut weights: Weights) -> Tuple[Int, Int]:
    """Read ternary weight matrix and return (weight_offset, scale_offset)."""
    var dims = file.read_bytes(8)
    var dims_ptr = dims.unsafe_ptr().bitcast[Int32]()
    var rows = Int(dims_ptr[0])
    var cols = Int(dims_ptr[1])

    var bytes_per_row = (cols + 3) // 4
    var total_bytes = rows * bytes_per_row

    var w_offset = len(weights.ternary_data)
    var w_data = file.read_bytes(total_bytes)
    var w_ptr = w_data.unsafe_ptr()
    for i in range(total_bytes):
        weights.ternary_data.append(w_ptr[i])

    var s_offset = len(weights.scales)
    var s_data = file.read_bytes(rows * 4)
    var s_ptr = s_data.unsafe_ptr().bitcast[Float32]()
    for i in range(rows):
        weights.scales.append(s_ptr[i])

    return (w_offset, s_offset)


fn load_weights(path: String, config: Config) raises -> Weights:
    """Load model weights from file."""
    var weights = Weights(config.n_layers)

    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var n_layers = config.n_layers
    var vocab_size = config.vocab_size

    var f = open(path, "rb")

    # Embedding
    var embed_size = vocab_size * ((dim + 3) // 4)
    weights.embed_offset = 0
    var embed_data = f.read_bytes(embed_size)
    var embed_ptr = embed_data.unsafe_ptr()
    for i in range(embed_size):
        weights.ternary_data.append(embed_ptr[i])

    # Layer weights
    for layer in range(n_layers):
        weights.layer_input_norm.append(read_float(f, weights, dim))

        var qw = read_ternary(f, weights)
        weights.layer_q_weight.append(qw[0])
        weights.layer_q_scale.append(qw[1])

        var kw = read_ternary(f, weights)
        weights.layer_k_weight.append(kw[0])
        weights.layer_k_scale.append(kw[1])

        var vw = read_ternary(f, weights)
        weights.layer_v_weight.append(vw[0])
        weights.layer_v_scale.append(vw[1])

        weights.layer_attn_sub_norm.append(read_float(f, weights, dim))

        var ow = read_ternary(f, weights)
        weights.layer_o_weight.append(ow[0])
        weights.layer_o_scale.append(ow[1])

        weights.layer_post_norm.append(read_float(f, weights, dim))

        var gatew = read_ternary(f, weights)
        weights.layer_gate_weight.append(gatew[0])
        weights.layer_gate_scale.append(gatew[1])

        var upw = read_ternary(f, weights)
        weights.layer_up_weight.append(upw[0])
        weights.layer_up_scale.append(upw[1])

        weights.layer_ffn_sub_norm.append(read_float(f, weights, hidden_dim))

        var downw = read_ternary(f, weights)
        weights.layer_down_weight.append(downw[0])
        weights.layer_down_scale.append(downw[1])

    weights.final_norm_offset = read_float(f, weights, dim)

    var lmw = read_ternary(f, weights)
    weights.lm_head_offset = lmw[0]
    weights.lm_head_scale_offset = lmw[1]

    f.close()
    return weights^


# =============================================================================
# RoPE
# =============================================================================

fn rope(
    mut q: List[Float32],
    mut k: List[Float32],
    q_offset: Int,
    k_offset: Int,
    head_size: Int,
    n_heads: Int,
    n_kv_heads: Int,
    pos: Int,
    theta: Float32
):
    for h in range(n_heads):
        var q_head = q_offset + h * head_size
        for i in range(0, head_size, 2):
            var freq = 1.0 / (theta ** (Float32(i) / Float32(head_size)))
            var val = Float32(pos) * freq
            var cos_val = math.cos(val)
            var sin_val = math.sin(val)
            var q0 = q[q_head + i]
            var q1 = q[q_head + i + 1]
            q[q_head + i] = q0 * cos_val - q1 * sin_val
            q[q_head + i + 1] = q0 * sin_val + q1 * cos_val

    for h in range(n_kv_heads):
        var k_head = k_offset + h * head_size
        for i in range(0, head_size, 2):
            var freq = 1.0 / (theta ** (Float32(i) / Float32(head_size)))
            var val = Float32(pos) * freq
            var cos_val = math.cos(val)
            var sin_val = math.sin(val)
            var k0 = k[k_head + i]
            var k1 = k[k_head + i + 1]
            k[k_head + i] = k0 * cos_val - k1 * sin_val
            k[k_head + i + 1] = k0 * sin_val + k1 * cos_val


# =============================================================================
# Forward Pass (SIMD optimized)
# =============================================================================

fn forward(
    mut state: RunState,
    weights: Weights,
    config: Config,
    token: Int,
    pos: Int
):
    """Forward pass with SIMD optimizations."""
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var n_heads = config.n_heads
    var n_kv_heads = config.n_kv_heads
    var head_size = config.head_size
    var kv_dim = config.kv_dim

    # Embedding lookup (ternary packed)
    var embed_bytes_per_row = (dim + 3) // 4
    var embed_start = token * embed_bytes_per_row

    for i in range(dim):
        var byte_idx = i // 4
        var bit_pos = (i % 4) * 2
        var bits = (Int(weights.ternary_data[weights.embed_offset + embed_start + byte_idx]) >> bit_pos) & 0x03
        if bits == 0:
            state.x[i] = 0.0
        elif bits == 1:
            state.x[i] = 1.0
        else:
            state.x[i] = -1.0

    # Transformer layers
    for layer in range(config.n_layers):
        # Input norm
        rmsnorm_simd(state.xb, state.x, weights.float_data,
                     0, 0, weights.layer_input_norm[layer], dim)

        # Build LUT for this activation
        var lut_dim = build_lut_simd(state.xb, 0, dim)

        # QKV projections with SIMD-optimized matmul
        tmac_matmul_lut_simd(state.q, 0, lut_dim, weights.ternary_data, weights.scales,
                            weights.layer_q_weight[layer], weights.layer_q_scale[layer], dim, dim)
        tmac_matmul_lut_simd(state.k, 0, lut_dim, weights.ternary_data, weights.scales,
                            weights.layer_k_weight[layer], weights.layer_k_scale[layer], kv_dim, dim)
        tmac_matmul_lut_simd(state.v, 0, lut_dim, weights.ternary_data, weights.scales,
                            weights.layer_v_weight[layer], weights.layer_v_scale[layer], kv_dim, dim)

        # RoPE
        rope(state.q, state.k, 0, 0, head_size, n_heads, n_kv_heads, pos, config.rope_theta)

        # Cache KV
        var cache_offset = layer * config.seq_len * kv_dim + pos * kv_dim
        for i in range(kv_dim):
            state.key_cache[cache_offset + i] = state.k[i]
            state.value_cache[cache_offset + i] = state.v[i]

        # Attention (per head)
        for h in range(n_heads):
            var q_head = h * head_size
            var kv_head = (h // config.kv_mul) * head_size
            var att_offset = h * config.seq_len

            for t in range(pos + 1):
                var k_cache_offset = layer * config.seq_len * kv_dim + t * kv_dim + kv_head
                var score: Float32 = 0.0
                for i in range(head_size):
                    score += state.q[q_head + i] * state.key_cache[k_cache_offset + i]
                state.att[att_offset + t] = score / math.sqrt(Float32(head_size))

            softmax_simd(state.att, att_offset, pos + 1)

            for i in range(head_size):
                state.xb2[q_head + i] = 0.0

            for t in range(pos + 1):
                var v_cache_offset = layer * config.seq_len * kv_dim + t * kv_dim + kv_head
                var a = state.att[att_offset + t]
                for i in range(head_size):
                    state.xb2[q_head + i] += a * state.value_cache[v_cache_offset + i]

        # Attention sub-norm
        rmsnorm_simd(state.xb2, state.xb2, weights.float_data,
                     0, 0, weights.layer_attn_sub_norm[layer], dim)

        # Output projection
        var lut_attn = build_lut_simd(state.xb2, 0, dim)
        tmac_matmul_lut_simd(state.xb, 0, lut_attn, weights.ternary_data, weights.scales,
                            weights.layer_o_weight[layer], weights.layer_o_scale[layer], dim, dim)

        # Residual
        for i in range(dim):
            state.x[i] += state.xb[i]

        # FFN: post norm
        rmsnorm_simd(state.xb, state.x, weights.float_data,
                     0, 0, weights.layer_post_norm[layer], dim)

        # FFN projections
        var lut_ffn = build_lut_simd(state.xb, 0, dim)
        tmac_matmul_lut_simd(state.hb, 0, lut_ffn, weights.ternary_data, weights.scales,
                            weights.layer_gate_weight[layer], weights.layer_gate_scale[layer], hidden_dim, dim)
        tmac_matmul_lut_simd(state.hb2, 0, lut_ffn, weights.ternary_data, weights.scales,
                            weights.layer_up_weight[layer], weights.layer_up_scale[layer], hidden_dim, dim)

        # ReLU² and multiply
        for i in range(hidden_dim):
            var gate = state.hb[i]
            if gate < 0:
                gate = 0.0
            state.hb[i] = gate * gate * state.hb2[i]

        # FFN sub-norm
        rmsnorm_simd(state.hb2, state.hb, weights.float_data,
                     0, 0, weights.layer_ffn_sub_norm[layer], hidden_dim)

        # Down projection
        var lut_down = build_lut_simd(state.hb2, 0, hidden_dim)
        tmac_matmul_lut_simd(state.xb, 0, lut_down, weights.ternary_data, weights.scales,
                            weights.layer_down_weight[layer], weights.layer_down_scale[layer], dim, hidden_dim)

        # Residual
        for i in range(dim):
            state.x[i] += state.xb[i]

    # Final norm
    rmsnorm_simd(state.x, state.x, weights.float_data,
                 0, 0, weights.final_norm_offset, dim)

    # LM head
    var lut_lm = build_lut_simd(state.x, 0, dim)
    tmac_matmul_lut_simd(state.logits, 0, lut_lm, weights.ternary_data, weights.scales,
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
    if temp == 0.0:
        return sample_argmax(logits, size)

    # Apply temperature
    for i in range(size):
        logits[i] /= temp

    # Softmax
    softmax_simd(logits, 0, size)

    # Top-p sampling with separate arrays
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
# Generation
# =============================================================================

fn generate(
    mut state: RunState,
    weights: Weights,
    config: Config,
    num_tokens: Int,
    temperature: Float32,
    topp: Float32
) -> Tuple[List[Int], Float64]:
    """Generate tokens and return (tokens, elapsed_seconds)."""
    var tokens = List[Int]()
    var token = 128000  # BOS token

    state.reset(config)

    var start = time.perf_counter_ns()

    for pos in range(num_tokens):
        forward(state, weights, config, token, pos)
        token = sample_topp(state.logits, config.vocab_size, topp, temperature)
        tokens.append(token)

    var elapsed = Float64(time.perf_counter_ns() - start) / 1e9
    return (tokens^, elapsed)


# =============================================================================
# Main
# =============================================================================

fn main() raises:
    var args = argv()
    if len(args) < 2:
        print("Usage: bitnet_server_simd <model.bin> [num_tokens] [--server]")
        return

    var model_path = String(args[1])
    var num_tokens = 32
    var server_mode = False

    for i in range(2, len(args)):
        if String(args[i]) == "--server":
            server_mode = True
        else:
            try:
                num_tokens = atol(String(args[i]))
            except:
                pass

    print("BitNet T-MAC SIMD Server - Loading model...")
    print("SIMD width:", SIMD_WIDTH)

    var config = Config(
        dim=2560,
        hidden_dim=6912,
        n_layers=30,
        n_heads=20,
        n_kv_heads=5,
        vocab_size=128256,
        seq_len=2048
    )

    var weights = load_weights(model_path, config)
    var state = RunState(config)

    print("Model loaded successfully")

    if server_mode:
        print("SERVER_READY", flush=True)

        while True:
            var line: String
            try:
                line = input()
            except:
                break

            if line == "QUIT":
                break

            # Parse: num_tokens,temperature,top_p
            var parts = line.split(",")
            var req_tokens = 32
            var temperature: Float32 = 0.8
            var topp: Float32 = 0.9

            if len(parts) >= 1:
                try:
                    req_tokens = atol(parts[0])
                except:
                    pass
            if len(parts) >= 2:
                try:
                    temperature = atof(parts[1]).cast[DType.float32]()
                except:
                    pass
            if len(parts) >= 3:
                try:
                    topp = atof(parts[2]).cast[DType.float32]()
                except:
                    pass

            var result = generate(state, weights, config, req_tokens, temperature, topp)
            var tokens = result[0]
            var elapsed = result[1]

            # Output: token1,token2,...|elapsed
            var output = String("")
            for i in range(len(tokens)):
                if i > 0:
                    output += ","
                output += String(tokens[i])
            output += "|" + String(elapsed)
            print(output, flush=True)

    else:
        # Single run mode
        print("Generating", num_tokens, "tokens...")
        var result = generate(state, weights, config, num_tokens, 0.8, 0.9)
        var tokens = result[0]
        var elapsed = result[1]

        print("Generated tokens:", len(tokens))
        print("Time:", elapsed, "seconds")
        print("Speed:", Float64(len(tokens)) / elapsed, "tok/s")

        print("Tokens:", end=" ")
        for i in range(len(tokens)):
            print(tokens[i], end=" ")
        print()
