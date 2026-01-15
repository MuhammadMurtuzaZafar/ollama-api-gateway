"""
T-MAC: Table Lookup-Based LLM Inference in Pure Mojo

Based on: "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge"
Paper: https://arxiv.org/abs/2407.00088

Key Innovation:
- NO MULTIPLICATION in matmul - only table lookups and additions
- Ternary weights {-1, 0, +1} stored as 2 bits per weight
- Precompute partial sums for groups of activations
- Use weight bits as indices into lookup tables

Memory: ~1.4GB for 7B model (vs 14GB FP16, vs 3.5GB Q4)
Speed: Up to 6.6x faster than llama.cpp on CPU
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
comptime LUT_GROUP_SIZE: Int = 4  # Group 4 weights → 2^8=256 LUT entries
comptime TERNARY_BITS: Int = 2    # 2 bits per ternary weight


# =============================================================================
# Ternary Weight Storage
# =============================================================================

struct TernaryMatrix:
    """
    Stores ternary weights {-1, 0, +1} using 2 bits per weight.
    Encoding: 00 = 0, 01 = +1, 11 = -1 (10 unused)
    Packs 4 weights per byte.
    """
    var data: List[UInt8]
    var num_weights: Int
    var rows: Int
    var cols: Int

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.num_weights = rows * cols
        # 4 weights per byte (2 bits each)
        var num_bytes = (self.num_weights + 3) // 4
        self.data = List[UInt8](capacity=num_bytes)
        for _ in range(num_bytes):
            self.data.append(0)

    fn __init__(out self, var data: List[UInt8], rows: Int, cols: Int):
        self.data = data^
        self.rows = rows
        self.cols = cols
        self.num_weights = rows * cols

    @always_inline
    fn get_weight(self, idx: Int) -> Int:
        """Extract single ternary weight: returns -1, 0, or +1."""
        var byte_idx = idx // 4
        var bit_offset = (idx % 4) * 2
        var bits = (Int(self.data[byte_idx]) >> bit_offset) & 0x03
        # Decode: 00=0, 01=+1, 11=-1
        if bits == 0:
            return 0
        elif bits == 1:
            return 1
        else:  # bits == 3 (or 2, treat as -1)
            return -1

    @always_inline
    fn get_group_index(self, base_idx: Int) -> Int:
        """
        Get 8-bit index for a group of 4 ternary weights.
        This index directly addresses the precomputed LUT.
        """
        var byte_idx = base_idx // 4
        return Int(self.data[byte_idx])

    fn set_weight(mut self, idx: Int, value: Int):
        """Set a ternary weight (-1, 0, or +1)."""
        var byte_idx = idx // 4
        var bit_offset = (idx % 4) * 2
        var bits: UInt8
        if value == 0:
            bits = 0
        elif value == 1:
            bits = 1
        else:  # value == -1
            bits = 3

        # Clear existing bits and set new value
        var mask = ~(UInt8(0x03) << bit_offset)
        self.data[byte_idx] = (self.data[byte_idx] & mask) | (bits << bit_offset)


# =============================================================================
# Lookup Table Engine
# =============================================================================

struct LookupTable:
    """
    Precomputed partial sums for activation groups.
    For 4 ternary weights (2 bits each), we have 256 possible combinations.
    Each entry stores the sum: sign_i times activation_i.
    """
    var tables: List[Float32]  # Flat storage: [num_groups × 256]
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
        """Get precomputed sum for a weight pattern."""
        return self.tables[group * 256 + index]

    @always_inline
    fn set(mut self, group: Int, index: Int, value: Float32):
        """Set precomputed sum for a weight pattern."""
        self.tables[group * 256 + index] = value


fn build_lut(activations: List[Float32], offset: Int, size: Int) -> LookupTable:
    """
    Build lookup table for a vector of activations.
    Groups activations by 4 and precomputes all 256 possible sums.

    For each group of 4 activations [a0, a1, a2, a3]:
    For each 8-bit pattern (encoding 4 ternary weights):
        LUT[pattern] = Σ decode(pattern, i) × a_i

    This is the KEY innovation - we do this ONCE per forward pass,
    then matmul becomes pure table lookups!
    """
    var num_groups = (size + 3) // 4
    var lut = LookupTable(num_groups)
    var act_ptr = activations.unsafe_ptr()

    for g in range(num_groups):
        var base = offset + g * 4

        # Get activations for this group (pad with 0 if needed)
        var a0 = act_ptr[base] if base < offset + size else Float32(0)
        var a1 = act_ptr[base + 1] if base + 1 < offset + size else Float32(0)
        var a2 = act_ptr[base + 2] if base + 2 < offset + size else Float32(0)
        var a3 = act_ptr[base + 3] if base + 3 < offset + size else Float32(0)

        # Precompute all 256 possible sums
        for pattern in range(256):
            # Decode 4 ternary weights from 8-bit pattern
            # Each weight uses 2 bits: 00=0, 01=+1, 11=-1
            var w0 = _decode_ternary((pattern >> 0) & 0x03)
            var w1 = _decode_ternary((pattern >> 2) & 0x03)
            var w2 = _decode_ternary((pattern >> 4) & 0x03)
            var w3 = _decode_ternary((pattern >> 6) & 0x03)

            # Compute partial sum (NO MULTIPLICATION of weights × activations!)
            # Just conditional adds/subtracts based on weight sign
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


@always_inline
fn _decode_ternary(bits: Int) -> Int:
    """Decode 2-bit pattern to ternary: 00=0, 01=+1, 11=-1."""
    if bits == 0:
        return 0
    elif bits == 1:
        return 1
    else:
        return -1


# =============================================================================
# T-MAC Matrix Multiplication (NO MULTIPLICATION!)
# =============================================================================

fn tmac_matmul(
    mut output: List[Float32],
    out_offset: Int,
    lut: LookupTable,
    weights: TernaryMatrix,
    w_row_offset: Int,
    rows: Int,
    cols: Int
):
    """
    T-MAC matrix-vector multiplication using lookup tables.

    Traditional: output[i] = Σ_j weight[i,j] × activation[j]  (expensive!)
    T-MAC:       output[i] = Σ_g LUT[g, weight_pattern[i,g]]  (just lookups!)

    This is the core innovation - we've precomputed all possible partial sums
    in the LUT, so matmul becomes pure table lookups and additions.
    """
    var w_ptr = weights.data.unsafe_ptr()
    var num_groups = (cols + 3) // 4

    for row in range(rows):
        var sum: Float32 = 0.0
        var w_base = (w_row_offset + row * cols) // 4  # Byte offset for this row

        # Sum across all groups - just lookups, no multiplications!
        for g in range(num_groups):
            var pattern = Int(w_ptr[w_base + g])
            sum += lut.get(g, pattern)

        output[out_offset + row] = sum


fn tmac_matmul_parallel(
    mut output: List[Float32],
    out_offset: Int,
    lut: LookupTable,
    weights: TernaryMatrix,
    w_row_offset: Int,
    rows: Int,
    cols: Int
):
    """Parallel T-MAC matmul for large matrices."""
    var w_ptr = weights.data.unsafe_ptr()
    var num_groups = (cols + 3) // 4

    @parameter
    fn compute_row(row: Int):
        var sum: Float32 = 0.0
        var w_base = (w_row_offset + row * cols) // 4

        for g in range(num_groups):
            var pattern = Int(w_ptr[w_base + g])
            sum += lut.get(g, pattern)

        output[out_offset + row] = sum

    parallelize[compute_row](rows)


# =============================================================================
# Float32 Matrix (for activations and non-quantized weights)
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

    fn __init__(out self, depth: Int, rows: Int, cols: Int):
        var size = depth * rows * cols
        self.data = List[Float32](capacity=size)
        for _ in range(size):
            self.data.append(0.0)
        self.depth = depth
        self.rows = rows
        self.cols = cols

    fn __init__(out self, var data: List[Float32], rows: Int, cols: Int = 1, depth: Int = 1):
        self.data = data^
        self.rows = rows
        self.cols = cols
        self.depth = depth

    @always_inline
    fn __getitem__(self, i: Int) -> Float32:
        return self.data[i]

    @always_inline
    fn __setitem__(mut self, i: Int, val: Float32):
        self.data[i] = val

    fn zero(mut self):
        for i in range(len(self.data)):
            self.data[i] = 0.0


# =============================================================================
# SIMD Operations for Non-LUT Components
# =============================================================================

comptime SIMD_W: Int = 8

fn rmsnorm_simd(mut out_data: List[Float32], out_offset: Int,
                x_data: List[Float32], x_offset: Int,
                w_data: List[Float32], w_offset: Int, size: Int):
    var x_ptr = x_data.unsafe_ptr()
    var w_ptr = w_data.unsafe_ptr()

    var ss: Float32 = 0.0
    var i = 0
    while i + SIMD_W <= size:
        var v = x_ptr.load[width=SIMD_W](x_offset + i)
        ss += (v * v).reduce_add()
        i += SIMD_W
    while i < size:
        var v = x_ptr[x_offset + i]
        ss += v * v
        i += 1

    ss = 1.0 / math.sqrt(ss / size + 1e-5)

    i = 0
    while i + SIMD_W <= size:
        var x = x_ptr.load[width=SIMD_W](x_offset + i)
        var w = w_ptr.load[width=SIMD_W](w_offset + i)
        var result = w * x * ss
        for j in range(SIMD_W):
            out_data[out_offset + i + j] = result[j]
        i += SIMD_W
    while i < size:
        out_data[out_offset + i] = w_ptr[w_offset + i] * x_ptr[x_offset + i] * ss
        i += 1


fn softmax(mut data: List[Float32], offset: Int, size: Int):
    var ptr = data.unsafe_ptr()

    var max_val = ptr[offset]
    for i in range(1, size):
        if ptr[offset + i] > max_val:
            max_val = ptr[offset + i]

    var sum_exp: Float32 = 0.0
    for i in range(size):
        var v = math.exp(ptr[offset + i] - max_val)
        data[offset + i] = v
        sum_exp += v

    var inv_sum = 1.0 / sum_exp
    for i in range(size):
        data[offset + i] *= inv_sum


# =============================================================================
# Model Configuration
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

    fn __init__(out self, path: String, print_config: Bool = False) raises:
        var f = open(path, "r")

        # Check for TMAC magic number
        var magic_bytes = f.read_bytes(4)
        var magic = String("")
        for i in range(4):
            magic += chr(Int(magic_bytes[i]))

        if magic != "TMAC":
            raise Error("Invalid T-MAC model file. Expected TMAC magic number.")

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
            print("T-MAC Config: dim=", self.dim, "hidden_dim=", self.hidden_dim)
            print("              n_layers=", self.n_layers, "n_heads=", self.n_heads)
            print("              vocab_size=", self.vocab_size)
            print("              Cores:", num_performance_cores())
            print("              LUT Group Size:", LUT_GROUP_SIZE)


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
# T-MAC Transformer Weights
# =============================================================================

struct TMACWeights:
    """Weights with ternary quantization for matmul layers."""
    # Ternary quantized (large matrices)
    var token_embedding: TernaryMatrix
    var wq: TernaryMatrix
    var wk: TernaryMatrix
    var wv: TernaryMatrix
    var wo: TernaryMatrix
    var w1: TernaryMatrix
    var w2: TernaryMatrix
    var w3: TernaryMatrix
    var wcls: TernaryMatrix

    # Float32 (small matrices - normalization and position)
    var rms_att_weight: Matrix
    var rms_ffn_weight: Matrix
    var rms_final_weight: Matrix
    var freq_cis_real: Matrix
    var freq_cis_imag: Matrix

    fn __init__(out self, path: String, config: Config) raises:
        var f = open(path, "r")
        # Skip magic + config
        _ = f.read_bytes(4 + NUM_CONFIG_INT * 4)

        fn read_ternary(mut file: FileHandle, num_weights: Int) raises -> TernaryMatrix:
            """Read ternary quantized weights (2 bits per weight)."""
            var num_bytes = (num_weights + 3) // 4
            var bytes = file.read_bytes(num_bytes)
            var data = List[UInt8](capacity=num_bytes)
            for i in range(num_bytes):
                data.append(bytes[i])
            # Calculate rows/cols from num_weights (assume square-ish)
            return TernaryMatrix(data^, 1, num_weights)

        fn read_float32(mut file: FileHandle, size: Int) raises -> List[Float32]:
            """Read float32 weights."""
            var bytes = file.read_bytes(size * 4)
            var ptr = bytes.unsafe_ptr().bitcast[Float32]()
            var result = List[Float32](capacity=size)
            for i in range(size):
                result.append(ptr[i])
            return result^

        # Read weights in order
        self.token_embedding = read_ternary(f, config.vocab_size * config.dim)
        self.rms_att_weight = Matrix(read_float32(f, config.n_layers * config.dim), config.n_layers, config.dim)
        self.wq = read_ternary(f, config.n_layers * config.dim * config.dim)
        self.wk = read_ternary(f, config.n_layers * config.kv_dim * config.dim)
        self.wv = read_ternary(f, config.n_layers * config.kv_dim * config.dim)
        self.wo = read_ternary(f, config.n_layers * config.dim * config.dim)
        self.rms_ffn_weight = Matrix(read_float32(f, config.n_layers * config.dim), config.n_layers, config.dim)
        self.w1 = read_ternary(f, config.n_layers * config.hidden_dim * config.dim)
        self.w2 = read_ternary(f, config.n_layers * config.dim * config.hidden_dim)
        self.w3 = read_ternary(f, config.n_layers * config.hidden_dim * config.dim)
        self.rms_final_weight = Matrix(read_float32(f, config.dim), config.dim)
        self.freq_cis_real = Matrix(read_float32(f, config.seq_len * config.head_size // 2), config.seq_len, config.head_size // 2)
        self.freq_cis_imag = Matrix(read_float32(f, config.seq_len * config.head_size // 2), config.seq_len, config.head_size // 2)

        if config.shared_weights:
            self.wcls = TernaryMatrix(0, 0)
        else:
            self.wcls = read_ternary(f, config.vocab_size * config.dim)

        f.close()


# =============================================================================
# Run State
# =============================================================================

struct RunState:
    var x: Matrix
    var xb: Matrix
    var xb2: Matrix
    var hb: Matrix
    var hb2: Matrix
    var q: Matrix
    var att: Matrix
    var logits: Matrix
    var key_cache: Matrix
    var value_cache: Matrix

    fn __init__(out self, config: Config):
        self.x = Matrix(config.dim)
        self.xb = Matrix(config.dim)
        self.xb2 = Matrix(config.dim)
        self.hb = Matrix(config.hidden_dim)
        self.hb2 = Matrix(config.hidden_dim)
        self.q = Matrix(config.dim)
        self.att = Matrix(config.n_heads, config.seq_len)
        self.logits = Matrix(config.vocab_size)
        self.key_cache = Matrix(config.n_layers, config.seq_len, config.kv_dim)
        self.value_cache = Matrix(config.n_layers, config.seq_len, config.kv_dim)


# =============================================================================
# T-MAC Transformer Forward Pass
# =============================================================================

fn get_token_embedding_tmac(
    mut out_data: List[Float32],
    out_offset: Int,
    token: Int,
    dim: Int,
    weights: TernaryMatrix
):
    """Extract and dequantize token embedding."""
    var weight_start = token * dim
    for i in range(dim):
        var w = weights.get_weight(weight_start + i)
        out_data[out_offset + i] = Float32(w)  # -1, 0, or +1


fn transformer_forward_tmac(
    token: Int,
    pos: Int,
    config: Config,
    mut state: RunState,
    weights: TMACWeights,
) raises:
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var head_size = config.head_size
    var kv_dim = config.kv_dim
    var kv_mul = config.kv_mul
    var n_heads = config.n_heads
    var n_kv_heads = config.n_kv_heads
    var sqrt_head_size = math.sqrt(Float32(head_size))

    # Token embedding
    get_token_embedding_tmac(state.x.data, 0, token, dim, weights.token_embedding)

    var freq_offset = pos * (head_size // 2)

    for layer in range(config.n_layers):
        var layer_dim_offset = layer * dim

        # RMSNorm (float32)
        rmsnorm_simd(state.xb.data, 0, state.x.data, 0,
                     weights.rms_att_weight.data, layer_dim_offset, dim)

        # Build LUT for normalized input - this is the key T-MAC step!
        var lut_xb = build_lut(state.xb.data, 0, dim)

        # Q, K, V projections using T-MAC (NO MULTIPLICATION!)
        var wq_offset = layer * dim * dim
        var wk_offset = layer * kv_dim * dim
        var wv_offset = layer * kv_dim * dim

        tmac_matmul_parallel(state.q.data, 0, lut_xb, weights.wq, wq_offset, dim, dim)

        var cache_offset = layer * config.seq_len * kv_dim + pos * kv_dim
        tmac_matmul(state.key_cache.data, cache_offset, lut_xb, weights.wk, wk_offset, kv_dim, dim)
        tmac_matmul(state.value_cache.data, cache_offset, lut_xb, weights.wv, wv_offset, kv_dim, dim)

        # RoPE (float32)
        var freq_real_ptr = weights.freq_cis_real.data.unsafe_ptr()
        var freq_imag_ptr = weights.freq_cis_imag.data.unsafe_ptr()
        for h in range(n_heads):
            for j in range(0, head_size, 2):
                var fcr = freq_real_ptr[freq_offset + j // 2]
                var fci = freq_imag_ptr[freq_offset + j // 2]

                var q_idx = h * head_size + j
                var q0 = state.q[q_idx]
                var q1 = state.q[q_idx + 1]
                state.q[q_idx] = q0 * fcr - q1 * fci
                state.q[q_idx + 1] = q0 * fci + q1 * fcr

                if h < n_kv_heads:
                    var k_idx = cache_offset + h * head_size + j
                    var k0 = state.key_cache[k_idx]
                    var k1 = state.key_cache[k_idx + 1]
                    state.key_cache[k_idx] = k0 * fcr - k1 * fci
                    state.key_cache[k_idx + 1] = k0 * fci + k1 * fcr

        state.xb.zero()

        # Attention (float32 - values already computed)
        var q_ptr = state.q.data.unsafe_ptr()
        var k_ptr = state.key_cache.data.unsafe_ptr()
        var v_ptr = state.value_cache.data.unsafe_ptr()

        for h in range(n_heads):
            var q_offset_h = h * head_size
            var att_offset = h * config.seq_len

            for t in range(pos + 1):
                var k_base = layer * config.seq_len * kv_dim + t * kv_dim + (h // kv_mul) * head_size

                var score: Float32 = 0.0
                for i in range(head_size):
                    score += q_ptr[q_offset_h + i] * k_ptr[k_base + i]

                state.att[att_offset + t] = score / sqrt_head_size

            softmax(state.att.data, att_offset, pos + 1)

            var xb_offset = h * head_size
            for t in range(pos + 1):
                var v_base = layer * config.seq_len * kv_dim + t * kv_dim + (h // kv_mul) * head_size
                var a = state.att[att_offset + t]

                for i in range(head_size):
                    state.xb[xb_offset + i] += a * v_ptr[v_base + i]

        # Output projection using T-MAC
        var lut_xb_out = build_lut(state.xb.data, 0, dim)
        var wo_offset = layer * dim * dim
        tmac_matmul_parallel(state.xb2.data, 0, lut_xb_out, weights.wo, wo_offset, dim, dim)

        # Residual
        for i in range(dim):
            state.x[i] += state.xb2[i]

        # FFN
        rmsnorm_simd(state.xb.data, 0, state.x.data, 0,
                     weights.rms_ffn_weight.data, layer_dim_offset, dim)

        # Build LUT for FFN input
        var lut_ffn = build_lut(state.xb.data, 0, dim)

        var w1_offset = layer * hidden_dim * dim
        var w3_offset = layer * hidden_dim * dim
        tmac_matmul_parallel(state.hb.data, 0, lut_ffn, weights.w1, w1_offset, hidden_dim, dim)
        tmac_matmul_parallel(state.hb2.data, 0, lut_ffn, weights.w3, w3_offset, hidden_dim, dim)

        # SiLU activation (float32)
        for i in range(hidden_dim):
            var v = state.hb[i]
            state.hb[i] = v * (1.0 / (1.0 + math.exp(-v))) * state.hb2[i]

        # Down projection using T-MAC
        var lut_hb = build_lut(state.hb.data, 0, hidden_dim)
        var w2_offset = layer * dim * hidden_dim
        tmac_matmul_parallel(state.xb.data, 0, lut_hb, weights.w2, w2_offset, dim, hidden_dim)

        # Residual
        for i in range(dim):
            state.x[i] += state.xb[i]

    # Final norm
    rmsnorm_simd(state.xb.data, 0, state.x.data, 0,
                 weights.rms_final_weight.data, 0, dim)

    # Classifier using T-MAC
    var lut_final = build_lut(state.xb.data, 0, dim)
    if config.shared_weights:
        tmac_matmul_parallel(state.logits.data, 0, lut_final, weights.token_embedding, 0, config.vocab_size, dim)
    else:
        tmac_matmul_parallel(state.logits.data, 0, lut_final, weights.wcls, 0, config.vocab_size, dim)


# =============================================================================
# Generation Utilities
# =============================================================================

fn argmax(data: List[Float32], size: Int) -> Int:
    var max_idx = 0
    var max_val = data[0]
    for i in range(1, size):
        if data[i] > max_val:
            max_val = data[i]
            max_idx = i
    return max_idx


fn sample(data: List[Float32], size: Int) -> Int:
    var r = random.random_float64().cast[DType.float32]()
    var cdf: Float32 = 0.0
    for i in range(size):
        cdf += data[i]
        if r < cdf:
            return i
    return size - 1


fn bpe_encode(mut tokens: List[Int], text: String, tok: Tokenizer):
    for i in range(len(text)):
        var c = String(text[i])
        var idx = tok.find(c)
        if idx == -1:
            return
        tokens.append(idx)

    while len(tokens) >= 2:
        var best_score = Float32(-1e10)
        var best_idx = -1
        var best_id = -1

        for i in range(len(tokens) - 1):
            var merged = tok.vocab[tokens[i]] + tok.vocab[tokens[i + 1]]
            var id = tok.find(merged)
            if id != -1 and tok.vocab_scores[id] > best_score:
                best_score = tok.vocab_scores[id]
                best_idx = i
                best_id = id

        if best_idx == -1:
            break

        tokens[best_idx] = best_id
        var new_tokens = List[Int]()
        for i in range(best_idx + 1):
            new_tokens.append(tokens[i])
        for i in range(best_idx + 2, len(tokens)):
            new_tokens.append(tokens[i])
        tokens = new_tokens^


fn print_token(tok: Tokenizer, token: Int):
    var s = tok.vocab[token]
    if s == "<0x0A>":
        print("\n", end="")
    else:
        print(s, end="")


# =============================================================================
# Main
# =============================================================================

fn main() raises:
    var checkpoint = "stories110M.tmac.bin"
    var tokenizer_path = "tokenizer.bin"
    var temperature: Float32 = 0.9
    var steps = 256
    var prompt = String("")

    var args = argv()
    if len(args) >= 2:
        checkpoint = args[1]

    var i = 2
    while i < len(args):
        if args[i] == "-z" and i + 1 < len(args):
            tokenizer_path = args[i + 1]
            i += 2
        elif args[i] == "-n" and i + 1 < len(args):
            steps = atol(args[i + 1])
            i += 2
        elif args[i] == "-t" and i + 1 < len(args):
            temperature = atof(args[i + 1]).cast[DType.float32]()
            i += 2
        elif args[i] == "-i" and i + 1 < len(args):
            prompt = args[i + 1]
            i += 2
        else:
            i += 1

    print("Loading T-MAC model from", checkpoint)
    print("T-MAC: Table Lookup-based Inference (NO MULTIPLICATION!)")
    var config = Config(checkpoint, True)
    var weights = TMACWeights(checkpoint, config)
    var tokenizer = Tokenizer(config.vocab_size, tokenizer_path)
    var state = RunState(config)

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    var prompt_tokens = List[Int]()
    if len(prompt) > 0:
        bpe_encode(prompt_tokens, prompt, tokenizer)

    print("Generating", steps, "tokens...")
    print("-" * 50)

    var token = 1
    var start_time = time.perf_counter_ns()
    var tokens_generated = 0

    for pos in range(steps):
        transformer_forward_tmac(token, pos, config, state, weights)

        var next_token: Int
        if pos < len(prompt_tokens):
            next_token = prompt_tokens[pos]
        else:
            if temperature == 0.0:
                next_token = argmax(state.logits.data, config.vocab_size)
            else:
                for j in range(config.vocab_size):
                    state.logits[j] /= temperature
                softmax(state.logits.data, 0, config.vocab_size)
                next_token = sample(state.logits.data, config.vocab_size)

        if next_token == 1 or next_token == 2:
            break

        print_token(tokenizer, next_token)
        token = next_token
        tokens_generated += 1

    var end_time = time.perf_counter_ns()
    var elapsed_ms = Int(end_time - start_time) // 1_000_000

    print("\n" + "-" * 50)
    print("Generated", tokens_generated, "tokens in", elapsed_ms, "ms")
    if elapsed_ms > 0:
        var tok_per_sec = tokens_generated * 1000 // elapsed_ms
        print("Speed:", tok_per_sec, "tokens/sec")
        print("Method: T-MAC (Lookup Table, NO MULTIPLICATION)")
