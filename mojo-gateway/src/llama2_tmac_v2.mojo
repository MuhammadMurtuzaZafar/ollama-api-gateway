"""
T-MAC v2: Table Lookup-Based LLM Inference with Per-Row Scales

Key improvement over v1: Per-row scale factors to preserve magnitude.
- Each row stores: [scale (float16)] + [ternary weights]
- Output = scale × LUT_result
- Better quality with ~12x compression (vs 16x for pure ternary)

Based on: "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment"
Paper: https://arxiv.org/abs/2407.00088
"""
from algorithm import parallelize
from collections import List, Dict
from memory import UnsafePointer
from sys import argv
from sys.info import num_performance_cores
import math
import random
import time

# Configuration
comptime NUM_CONFIG_INT: Int = 7
comptime SIMD_WIDTH: Int = 8
comptime LUT_GROUP_SIZE: Int = 4


# =============================================================================
# Scaled Ternary Matrix (with per-row scales)
# =============================================================================

struct ScaledTernaryMatrix:
    """
    Ternary weights with per-row scale factors.
    Format per row: [scale: float16 (2 bytes)] + [ternary: cols/4 bytes]
    Output = scale × ternary_dot_product
    """
    var data: List[UInt8]
    var scales: List[Float32]  # Scales stored as float32 for computation
    var rows: Int
    var cols: Int
    var bytes_per_row: Int

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.bytes_per_row = (cols + 3) // 4  # Ternary bytes only
        var total_ternary_bytes = rows * self.bytes_per_row
        self.data = List[UInt8](capacity=total_ternary_bytes)
        for _ in range(total_ternary_bytes):
            self.data.append(0)
        self.scales = List[Float32](capacity=rows)
        for _ in range(rows):
            self.scales.append(1.0)

    fn __init__(out self, var data: List[UInt8], var scales: List[Float32], rows: Int, cols: Int):
        self.data = data^
        self.scales = scales^
        self.rows = rows
        self.cols = cols
        self.bytes_per_row = (cols + 3) // 4

    @always_inline
    fn get_scale(self, row: Int) -> Float32:
        """Get the scale factor for a row."""
        return self.scales[row]

    @always_inline
    fn get_ternary_byte(self, row: Int, byte_idx: Int) -> UInt8:
        """Get a byte of packed ternary weights for a row."""
        return self.data[row * self.bytes_per_row + byte_idx]


# =============================================================================
# Lookup Table Engine
# =============================================================================

struct LookupTable:
    """
    Precomputed partial sums for activation groups.
    For 4 ternary weights (2 bits each), we have 256 possible combinations.
    """
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
    fn get(self, group: Int, index: Int) -> Float32:
        return self.tables[group * 256 + index]

    @always_inline
    fn set(mut self, group: Int, index: Int, value: Float32):
        self.tables[group * 256 + index] = value


@always_inline
fn _decode_ternary(bits: Int) -> Int:
    """Decode 2-bit pattern to ternary: 00=0, 01=+1, 11=-1."""
    if bits == 0:
        return 0
    elif bits == 1:
        return 1
    else:
        return -1


fn build_lut(activations: List[Float32], offset: Int, size: Int) -> LookupTable:
    """Build lookup table for a vector of activations."""
    var num_groups = (size + 3) // 4
    var lut = LookupTable(num_groups)
    var act_ptr = activations.unsafe_ptr()

    for g in range(num_groups):
        var base = offset + g * 4
        var a0 = act_ptr[base] if base < offset + size else Float32(0)
        var a1 = act_ptr[base + 1] if base + 1 < offset + size else Float32(0)
        var a2 = act_ptr[base + 2] if base + 2 < offset + size else Float32(0)
        var a3 = act_ptr[base + 3] if base + 3 < offset + size else Float32(0)

        # Precompute all 256 possible sums
        for pattern in range(256):
            var w0 = _decode_ternary((pattern >> 0) & 0x03)
            var w1 = _decode_ternary((pattern >> 2) & 0x03)
            var w2 = _decode_ternary((pattern >> 4) & 0x03)
            var w3 = _decode_ternary((pattern >> 6) & 0x03)

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
# T-MAC v2 Matrix Multiplication (with scales)
# =============================================================================

fn tmac_v2_matmul_parallel(
    mut output: List[Float32],
    out_offset: Int,
    lut: LookupTable,
    weights: ScaledTernaryMatrix,
    row_offset: Int,
    num_rows: Int,
):
    """
    T-MAC v2 matmul with per-row scaling.
    output[i] = scale[i] × Σ_g LUT[g, pattern[i,g]]
    """
    var w_ptr = weights.data.unsafe_ptr()
    var scale_ptr = weights.scales.unsafe_ptr()
    var bytes_per_row = weights.bytes_per_row
    var num_groups = bytes_per_row

    @parameter
    fn compute_row(row: Int):
        var actual_row = row_offset + row
        var sum: Float32 = 0.0
        var w_base = actual_row * bytes_per_row

        # LUT lookups
        for g in range(num_groups):
            var pattern = Int(w_ptr[w_base + g])
            sum += lut.get(g, pattern)

        # Apply per-row scale
        output[out_offset + row] = sum * scale_ptr[actual_row]

    parallelize[compute_row](num_rows)


# =============================================================================
# Float32 Matrix
# =============================================================================

struct Matrix:
    var data: List[Float32]
    var rows: Int
    var cols: Int
    var depth: Int

    fn __init__(out self, size: Int):
        self.data = List[Float32](capacity=size)
        for _ in range(size):
            self.data.append(0.0)
        self.rows = size
        self.cols = 1
        self.depth = 1

    fn __init__(out self, rows: Int, cols: Int):
        var size = rows * cols
        self.data = List[Float32](capacity=size)
        for _ in range(size):
            self.data.append(0.0)
        self.rows = rows
        self.cols = cols
        self.depth = 1

    fn __init__(out self, var data: List[Float32], rows: Int, cols: Int = 1, depth: Int = 1):
        self.data = data^
        self.rows = rows
        self.cols = cols
        self.depth = depth

    fn __init__(out self, var data: List[Float32], size: Int):
        self.data = data^
        self.rows = size
        self.cols = 1
        self.depth = 1


# =============================================================================
# SIMD Operations
# =============================================================================

@always_inline
fn rmsnorm(mut output: List[Float32], input: List[Float32], weight: Matrix,
           o_offset: Int, i_offset: Int, w_offset: Int, size: Int):
    var ss: Float32 = 0.0
    for i in range(size):
        ss += input[i_offset + i] * input[i_offset + i]
    ss = 1.0 / math.sqrt(ss / Float32(size) + 1e-5)
    for i in range(size):
        output[o_offset + i] = weight.data[w_offset + i] * (ss * input[i_offset + i])


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


fn matmul_parallel(mut output: List[Float32], input: List[Float32], weight: Matrix,
                   o_offset: Int, i_offset: Int, w_offset: Int,
                   rows: Int, cols: Int):
    """Standard float32 matmul for non-quantized weights."""
    var w_ptr = weight.data.unsafe_ptr()
    var i_ptr = input.unsafe_ptr()

    @parameter
    fn compute_row(row: Int):
        var sum: Float32 = 0.0
        var base = w_offset + row * cols
        for j in range(cols):
            sum += w_ptr[base + j] * i_ptr[i_offset + j]
        output[o_offset + row] = sum

    parallelize[compute_row](rows)


# =============================================================================
# Config
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
    var shared_weights: Bool

    fn __init__(out self, path: String, print_config: Bool = True) raises:
        var f = open(path, "r")

        # Check for TM2 magic
        var magic_bytes = f.read_bytes(4)
        var magic = String("")
        for i in range(3):
            magic += chr(Int(magic_bytes[i]))

        if magic != "TM2":
            raise Error("Invalid T-MAC v2 model. Expected TM2 magic number.")

        var config_bytes = f.read_bytes(NUM_CONFIG_INT * 4)
        f.close()

        var ptr = config_bytes.unsafe_ptr().bitcast[Int32]()
        self.dim = Int(ptr[0])
        self.hidden_dim = Int(ptr[1])
        self.n_layers = Int(ptr[2])
        self.n_heads = Int(ptr[3])
        self.n_kv_heads = Int(ptr[4])
        self.vocab_size = Int(ptr[5])
        self.seq_len = Int(ptr[6])

        self.head_size = self.dim // self.n_heads
        self.kv_dim = (self.n_kv_heads * self.dim) // self.n_heads
        self.kv_mul = self.n_heads // self.n_kv_heads
        self.shared_weights = self.vocab_size > 0

        if not self.shared_weights:
            self.vocab_size = -self.vocab_size

        if print_config:
            print("T-MAC v2 Config: dim=", self.dim, "hidden_dim=", self.hidden_dim)
            print("                 n_layers=", self.n_layers, "n_heads=", self.n_heads)
            print("                 vocab_size=", self.vocab_size)
            print("                 Cores:", num_performance_cores())
            print("                 Per-row scaling: ENABLED")


# =============================================================================
# Tokenizer
# =============================================================================

struct Tokenizer:
    var vocab: List[String]
    var vocab_scores: List[Float32]
    var vocab_size: Int
    var vocab_map: Dict[String, Int]

    fn __init__(out self, vocab_size: Int, path: String) raises:
        self.vocab_size = vocab_size
        self.vocab = List[String]()
        self.vocab_scores = List[Float32]()
        self.vocab_map = Dict[String, Int]()

        var f = open(path, "r")
        _ = f.read_bytes(4)

        for i in range(vocab_size):
            var score_bytes = f.read_bytes(4)
            var score = score_bytes.unsafe_ptr().bitcast[Float32]()[0]
            self.vocab_scores.append(score)

            var len_bytes = f.read_bytes(4)
            var token_len = Int(len_bytes.unsafe_ptr().bitcast[Int32]()[0])

            var token_bytes = f.read_bytes(token_len)
            var token = String("")
            for j in range(token_len):
                token += chr(Int(token_bytes[j]))

            self.vocab.append(token)
            self.vocab_map[token] = i

        f.close()

    fn find(self, token: String) -> Int:
        var result = self.vocab_map.find(token)
        if result:
            return result.value()
        return -1


# =============================================================================
# T-MAC v2 Weights
# =============================================================================

struct TMACv2Weights:
    """Weights with scaled ternary quantization."""
    var token_embedding: ScaledTernaryMatrix
    var wq: ScaledTernaryMatrix
    var wk: ScaledTernaryMatrix
    var wv: ScaledTernaryMatrix
    var wo: ScaledTernaryMatrix
    var w1: ScaledTernaryMatrix
    var w2: ScaledTernaryMatrix
    var w3: ScaledTernaryMatrix
    var wcls: ScaledTernaryMatrix

    var rms_att_weight: Matrix
    var rms_ffn_weight: Matrix
    var rms_final_weight: Matrix
    var freq_cis_real: Matrix
    var freq_cis_imag: Matrix

    fn __init__(out self, path: String, config: Config) raises:
        var f = open(path, "r")
        # Skip magic (4) + config (28)
        _ = f.read_bytes(4 + NUM_CONFIG_INT * 4)

        fn read_scaled_ternary(mut file: FileHandle, rows: Int, cols: Int) raises -> ScaledTernaryMatrix:
            """Read scaled ternary matrix: flag + dims + [scale + ternary_row] per row."""
            var flag_byte = file.read_bytes(1)
            var flag = Int(flag_byte[0])
            if flag != 1:
                raise Error("Expected quantized weight flag")

            var dims_bytes = file.read_bytes(8)
            var dims_ptr = dims_bytes.unsafe_ptr().bitcast[Int32]()
            var file_rows = Int(dims_ptr[0])
            var file_cols = Int(dims_ptr[1])

            var bytes_per_row = (file_cols + 3) // 4
            var scales = List[Float32](capacity=file_rows)
            var data = List[UInt8](capacity=file_rows * bytes_per_row)

            for row in range(file_rows):
                # Read scale (float16 -> float32)
                var scale_bytes = file.read_bytes(2)
                var scale_f16 = scale_bytes.unsafe_ptr().bitcast[Float16]()[0]
                scales.append(Float32(scale_f16))

                # Read ternary row
                var ternary_bytes = file.read_bytes(bytes_per_row)
                for b in range(bytes_per_row):
                    data.append(ternary_bytes[b])

            return ScaledTernaryMatrix(data^, scales^, file_rows, file_cols)

        fn read_float32(mut file: FileHandle, size: Int) raises -> List[Float32]:
            """Read float32 weights with flag."""
            var flag_byte = file.read_bytes(1)
            var bytes_data = file.read_bytes(size * 4)
            var ptr = bytes_data.unsafe_ptr().bitcast[Float32]()
            var result = List[Float32](capacity=size)
            for i in range(size):
                result.append(ptr[i])
            return result^

        # Read all weights - dimensions match Python quantizer
        self.token_embedding = read_scaled_ternary(f, config.vocab_size, config.dim)
        self.rms_att_weight = Matrix(read_float32(f, config.n_layers * config.dim), config.n_layers, config.dim)
        self.wq = read_scaled_ternary(f, config.n_layers * config.dim, config.dim)
        self.wk = read_scaled_ternary(f, config.n_layers * config.kv_dim, config.dim)
        self.wv = read_scaled_ternary(f, config.n_layers * config.kv_dim, config.dim)
        self.wo = read_scaled_ternary(f, config.n_layers * config.dim, config.dim)
        self.rms_ffn_weight = Matrix(read_float32(f, config.n_layers * config.dim), config.n_layers, config.dim)
        self.w1 = read_scaled_ternary(f, config.n_layers * config.hidden_dim, config.dim)
        self.w2 = read_scaled_ternary(f, config.n_layers * config.dim, config.hidden_dim)
        self.w3 = read_scaled_ternary(f, config.n_layers * config.hidden_dim, config.dim)
        self.rms_final_weight = Matrix(read_float32(f, config.dim), config.dim, 1)
        self.freq_cis_real = Matrix(read_float32(f, config.seq_len * config.head_size // 2), config.seq_len, config.head_size // 2)
        self.freq_cis_imag = Matrix(read_float32(f, config.seq_len * config.head_size // 2), config.seq_len, config.head_size // 2)

        if config.shared_weights:
            self.wcls = ScaledTernaryMatrix(0, 0)
        else:
            self.wcls = read_scaled_ternary(f, config.vocab_size, config.dim)

        f.close()


# =============================================================================
# Run State
# =============================================================================

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


# =============================================================================
# Transformer Forward Pass
# =============================================================================

fn transformer(
    token: Int,
    pos: Int,
    config: Config,
    mut state: RunState,
    weights: TMACv2Weights
):
    """Forward pass using T-MAC v2 (scaled ternary)."""
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var head_size = config.head_size
    var kv_dim = config.kv_dim
    var kv_mul = config.kv_mul
    var n_heads = config.n_heads
    var n_kv_heads = config.n_kv_heads
    var seq_len = config.seq_len

    # Get embedding (using scaled ternary)
    var emb_row = token
    var emb_scale = weights.token_embedding.get_scale(emb_row)
    var emb_bytes_per_row = weights.token_embedding.bytes_per_row
    for i in range(dim):
        var byte_idx = i // 4
        var bit_offset = (i % 4) * 2
        var byte_val = weights.token_embedding.get_ternary_byte(emb_row, byte_idx)
        var bits = (Int(byte_val) >> bit_offset) & 0x03
        var ternary = _decode_ternary(bits)
        state.x[i] = Float32(ternary) * emb_scale

    # Process each layer
    for layer in range(config.n_layers):
        # RMSNorm for attention
        rmsnorm(state.xb, state.x, weights.rms_att_weight, 0, 0, layer * dim, dim)

        # Build LUT for current activations
        var lut = build_lut(state.xb, 0, dim)

        # QKV projections using T-MAC v2
        # wq: (n_layers * dim, dim) -> dim rows per layer
        var q_row_offset = layer * dim
        tmac_v2_matmul_parallel(state.q, 0, lut, weights.wq, q_row_offset, dim)

        # wk/wv: (n_layers * kv_dim, dim) -> kv_dim rows per layer
        var k_row_offset = layer * kv_dim
        tmac_v2_matmul_parallel(state.k, 0, lut, weights.wk, k_row_offset, kv_dim)

        var v_row_offset = layer * kv_dim
        tmac_v2_matmul_parallel(state.v, 0, lut, weights.wv, v_row_offset, kv_dim)

        # RoPE
        for i in range(0, dim, 2):
            var head_dim = i % head_size
            var freq_idx = pos * (head_size // 2) + head_dim // 2
            var fcr = weights.freq_cis_real.data[freq_idx]
            var fci = weights.freq_cis_imag.data[freq_idx]
            var v0 = state.q[i]
            var v1 = state.q[i + 1]
            state.q[i] = v0 * fcr - v1 * fci
            state.q[i + 1] = v0 * fci + v1 * fcr

        for i in range(0, kv_dim, 2):
            var head_dim = i % head_size
            var freq_idx = pos * (head_size // 2) + head_dim // 2
            var fcr = weights.freq_cis_real.data[freq_idx]
            var fci = weights.freq_cis_imag.data[freq_idx]
            var v0 = state.k[i]
            var v1 = state.k[i + 1]
            state.k[i] = v0 * fcr - v1 * fci
            state.k[i + 1] = v0 * fci + v1 * fcr

        # Cache K and V
        var cache_offset = layer * seq_len * kv_dim + pos * kv_dim
        for i in range(kv_dim):
            state.key_cache[cache_offset + i] = state.k[i]
            state.value_cache[cache_offset + i] = state.v[i]

        # Attention
        for h in range(n_heads):
            var q_offset = h * head_size
            var kv_head = h // kv_mul

            # Compute attention scores
            for t in range(pos + 1):
                var k_offset = layer * seq_len * kv_dim + t * kv_dim + kv_head * head_size
                var score: Float32 = 0.0
                for i in range(head_size):
                    score += state.q[q_offset + i] * state.key_cache[k_offset + i]
                state.att[h * seq_len + t] = score / math.sqrt(Float32(head_size))

            # Softmax
            softmax(state.att, h * seq_len, pos + 1)

            # Weighted sum of values
            for i in range(head_size):
                state.xb[q_offset + i] = 0.0
            for t in range(pos + 1):
                var v_offset = layer * seq_len * kv_dim + t * kv_dim + kv_head * head_size
                var a = state.att[h * seq_len + t]
                for i in range(head_size):
                    state.xb[q_offset + i] += a * state.value_cache[v_offset + i]

        # Output projection
        var wo_lut = build_lut(state.xb, 0, dim)
        var wo_row_offset = layer * dim
        tmac_v2_matmul_parallel(state.xb2, 0, wo_lut, weights.wo, wo_row_offset, dim)

        # Residual
        for i in range(dim):
            state.x[i] += state.xb2[i]

        # FFN
        rmsnorm(state.xb, state.x, weights.rms_ffn_weight, 0, 0, layer * dim, dim)

        var ffn_lut = build_lut(state.xb, 0, dim)

        # w1 and w3 projections
        var w1_row_offset = layer * hidden_dim
        tmac_v2_matmul_parallel(state.hb, 0, ffn_lut, weights.w1, w1_row_offset, hidden_dim)

        var w3_row_offset = layer * hidden_dim
        tmac_v2_matmul_parallel(state.hb2, 0, ffn_lut, weights.w3, w3_row_offset, hidden_dim)

        # SiLU and element-wise multiply
        for i in range(hidden_dim):
            var val = state.hb[i]
            state.hb[i] = val * (1.0 / (1.0 + math.exp(-val))) * state.hb2[i]

        # w2 projection
        var w2_lut = build_lut(state.hb, 0, hidden_dim)
        var w2_row_offset = layer * dim
        tmac_v2_matmul_parallel(state.xb, 0, w2_lut, weights.w2, w2_row_offset, dim)

        # Residual
        for i in range(dim):
            state.x[i] += state.xb[i]

    # Final norm (copy to xb first to avoid aliasing)
    for i in range(dim):
        state.xb[i] = state.x[i]
    rmsnorm(state.x, state.xb, weights.rms_final_weight, 0, 0, 0, dim)

    # Classifier
    if config.shared_weights:
        var cls_lut = build_lut(state.x, 0, dim)
        tmac_v2_matmul_parallel(state.logits, 0, cls_lut, weights.token_embedding, 0, config.vocab_size)
    else:
        var cls_lut = build_lut(state.x, 0, dim)
        tmac_v2_matmul_parallel(state.logits, 0, cls_lut, weights.wcls, 0, config.vocab_size)


fn sample_argmax(logits: List[Float32], size: Int) -> Int:
    var max_idx = 0
    var max_val = logits[0]
    for i in range(1, size):
        if logits[i] > max_val:
            max_val = logits[i]
            max_idx = i
    return max_idx


fn sample_topp(logits: List[Float32], size: Int, topp: Float32, temp: Float32) -> Int:
    if temp == 0.0:
        return sample_argmax(logits, size)

    var probs = List[Float32](capacity=size)
    var max_val = logits[0]
    for i in range(1, size):
        if logits[i] > max_val:
            max_val = logits[i]

    var sum_exp: Float32 = 0.0
    for i in range(size):
        probs.append(math.exp((logits[i] - max_val) / temp))
        sum_exp += probs[i]
    for i in range(size):
        probs[i] /= sum_exp

    var indices = List[Int](capacity=size)
    for i in range(size):
        indices.append(i)

    for i in range(size):
        for j in range(i + 1, size):
            if probs[j] > probs[i]:
                var tmp = probs[i]
                probs[i] = probs[j]
                probs[j] = tmp
                var tmp_idx = indices[i]
                indices[i] = indices[j]
                indices[j] = tmp_idx

    var cumsum: Float32 = 0.0
    var cutoff = 0
    for i in range(size):
        cumsum += probs[i]
        if cumsum > topp:
            cutoff = i + 1
            break

    if cutoff == 0:
        cutoff = size

    var r = random.random_float64().cast[DType.float32]() * cumsum
    cumsum = 0.0
    for i in range(cutoff):
        cumsum += probs[i]
        if cumsum > r:
            return indices[i]

    return indices[cutoff - 1]


# =============================================================================
# Main
# =============================================================================

fn main() raises:
    var args = argv()
    if len(args) < 2:
        print("Usage: llama2_tmac_v2 <model.tmac2.bin> -z <tokenizer.bin> -n <tokens> -t <temp>")
        return

    var model_path = String(args[1])
    var tokenizer_path = String("tokenizer.bin")
    var num_tokens = 256
    var temperature: Float32 = 1.0
    var topp: Float32 = 0.9

    var i = 2
    while i < len(args):
        if String(args[i]) == "-z" and i + 1 < len(args):
            tokenizer_path = String(args[i + 1])
            i += 2
        elif String(args[i]) == "-n" and i + 1 < len(args):
            num_tokens = atol(args[i + 1])
            i += 2
        elif String(args[i]) == "-t" and i + 1 < len(args):
            temperature = atof(args[i + 1]).cast[DType.float32]()
            i += 2
        elif String(args[i]) == "-p" and i + 1 < len(args):
            topp = atof(args[i + 1]).cast[DType.float32]()
            i += 2
        else:
            i += 1

    print("Loading T-MAC v2 model from", model_path)
    print("T-MAC v2: Per-row scaled ternary (improved quality)")
    var config = Config(model_path)
    var weights = TMACv2Weights(model_path, config)
    var tokenizer = Tokenizer(config.vocab_size, tokenizer_path)
    var state = RunState(config)

    print("Generating", num_tokens, "tokens...")
    print("-" * 50)

    var token = 1  # BOS
    var start = time.perf_counter_ns()

    for pos in range(num_tokens):
        transformer(token, pos, config, state, weights)
        var next_token = sample_topp(state.logits, config.vocab_size, topp, temperature)

        if next_token == 1:
            break

        var token_str = tokenizer.vocab[next_token]
        print(token_str, end="")
        token = next_token

    var elapsed = (time.perf_counter_ns() - start) / 1_000_000

    print()
    print("-" * 50)
    print("Generated", num_tokens, "tokens in", Int(elapsed), "ms")
    print("Speed:", Int(Float32(num_tokens) / Float32(elapsed / 1000)), "tokens/sec")
    print("Method: T-MAC v2 (Per-row scaled, LUT-based)")
