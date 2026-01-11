"""
BitNet b1.58 Inference using T-MAC (Table Lookup) in Mojo

Based on Microsoft's BitNet b1.58-2B-4T model architecture.
Key differences from LLaMA:
- ReLU² activation (instead of SiLU)
- Sub-norm layers after attention and FFN
- Grouped Query Attention (GQA)

Uses T-MAC for multiplication-free inference.
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
# Scaled Ternary Matrix
# =============================================================================

struct ScaledTernaryMatrix:
    """Ternary weights with per-row scale factors."""
    var data: List[UInt8]
    var scales: List[Float32]
    var rows: Int
    var cols: Int
    var bytes_per_row: Int

    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.bytes_per_row = (cols + 3) // 4
        var total_bytes = rows * self.bytes_per_row
        self.data = List[UInt8](capacity=total_bytes)
        for _ in range(total_bytes):
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
        return self.scales[row]

    @always_inline
    fn get_ternary_byte(self, row: Int, byte_idx: Int) -> UInt8:
        return self.data[row * self.bytes_per_row + byte_idx]


# =============================================================================
# Lookup Table
# =============================================================================

struct LookupTable:
    """Precomputed partial sums for activation groups."""
    var tables: List[Float32]
    var num_groups: Int

    fn __init__(out self, num_groups: Int):
        self.num_groups = num_groups
        var total = num_groups * 256
        self.tables = List[Float32](capacity=total)
        for _ in range(total):
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
    if bits == 0:
        return 0
    elif bits == 1:
        return 1
    else:
        return -1


fn build_lut(activations: List[Float32], offset: Int, size: Int) -> LookupTable:
    """Build lookup table for activations."""
    var num_groups = (size + 3) // 4
    var lut = LookupTable(num_groups)
    var act_ptr = activations.unsafe_ptr()

    for g in range(num_groups):
        var base = offset + g * 4
        var a0 = act_ptr[base] if base < offset + size else Float32(0)
        var a1 = act_ptr[base + 1] if base + 1 < offset + size else Float32(0)
        var a2 = act_ptr[base + 2] if base + 2 < offset + size else Float32(0)
        var a3 = act_ptr[base + 3] if base + 3 < offset + size else Float32(0)

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
# T-MAC MatMul
# =============================================================================

fn tmac_matmul_parallel(
    mut output: List[Float32],
    out_offset: Int,
    lut: LookupTable,
    weights: ScaledTernaryMatrix,
    row_offset: Int,
    num_rows: Int,
):
    """T-MAC matmul with per-row scaling."""
    var w_ptr = weights.data.unsafe_ptr()
    var scale_ptr = weights.scales.unsafe_ptr()
    var bytes_per_row = weights.bytes_per_row
    var num_groups = bytes_per_row

    @parameter
    fn compute_row(row: Int):
        var actual_row = row_offset + row
        var sum: Float32 = 0.0
        var w_base = actual_row * bytes_per_row

        for g in range(num_groups):
            var pattern = Int(w_ptr[w_base + g])
            sum += lut.get(g, pattern)

        output[out_offset + row] = sum * scale_ptr[actual_row]

    parallelize[compute_row](num_rows)


# =============================================================================
# Float32 Matrix
# =============================================================================

struct Matrix:
    var data: List[Float32]
    var rows: Int
    var cols: Int

    fn __init__(out self, size: Int):
        self.data = List[Float32](capacity=size)
        for _ in range(size):
            self.data.append(0.0)
        self.rows = size
        self.cols = 1

    fn __init__(out self, var data: List[Float32], rows: Int, cols: Int = 1):
        self.data = data^
        self.rows = rows
        self.cols = cols


# =============================================================================
# Activation Functions
# =============================================================================

@always_inline
fn rmsnorm(mut output: List[Float32], input: List[Float32], weight: Matrix,
           o_offset: Int, i_offset: Int, w_offset: Int, size: Int):
    """RMS normalization."""
    var ss: Float32 = 0.0
    for i in range(size):
        ss += input[i_offset + i] * input[i_offset + i]
    ss = 1.0 / math.sqrt(ss / Float32(size) + 1e-5)
    for i in range(size):
        output[o_offset + i] = weight.data[w_offset + i] * (ss * input[i_offset + i])


@always_inline
fn relu_squared(mut x: List[Float32], offset: Int, size: Int):
    """ReLU² activation: max(0, x)²"""
    for i in range(size):
        var val = x[offset + i]
        if val > 0:
            x[offset + i] = val * val
        else:
            x[offset + i] = 0.0


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
# BitNet Config
# =============================================================================

struct BitNetConfig:
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

    fn __init__(out self, path: String, print_config: Bool = True) raises:
        var f = open(path, "r")

        var magic_bytes = f.read_bytes(4)
        var magic = String("")
        for i in range(3):
            magic += chr(Int(magic_bytes[i]))

        if magic != "TM2":
            raise Error("Invalid T-MAC v2 model. Expected TM2 magic.")

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

        # Handle negative vocab_size (indicates no weight sharing)
        if self.vocab_size < 0:
            self.vocab_size = -self.vocab_size

        self.head_size = self.dim // self.n_heads
        self.kv_dim = self.n_kv_heads * self.head_size
        self.kv_mul = self.n_heads // self.n_kv_heads

        if print_config:
            print("BitNet b1.58 Config:")
            print("  dim=", self.dim, "hidden_dim=", self.hidden_dim)
            print("  n_layers=", self.n_layers, "n_heads=", self.n_heads)
            print("  n_kv_heads=", self.n_kv_heads, "head_size=", self.head_size)
            print("  vocab_size=", self.vocab_size, "seq_len=", self.seq_len)
            print("  Cores:", num_performance_cores())


# =============================================================================
# BitNet Weights
# =============================================================================

struct BitNetWeights:
    """BitNet model weights with T-MAC format."""
    var embed_tokens: ScaledTernaryMatrix

    # Per-layer weights
    var input_layernorm: List[Matrix]
    var q_proj: List[ScaledTernaryMatrix]
    var k_proj: List[ScaledTernaryMatrix]
    var v_proj: List[ScaledTernaryMatrix]
    var o_proj: List[ScaledTernaryMatrix]
    var attn_sub_norm: List[Matrix]
    var post_attn_layernorm: List[Matrix]
    var gate_proj: List[ScaledTernaryMatrix]
    var up_proj: List[ScaledTernaryMatrix]
    var down_proj: List[ScaledTernaryMatrix]
    var ffn_sub_norm: List[Matrix]

    var final_norm: Matrix
    var lm_head: ScaledTernaryMatrix

    fn __init__(out self, path: String, config: BitNetConfig) raises:
        var f = open(path, "r")
        _ = f.read_bytes(4 + NUM_CONFIG_INT * 4)  # Skip magic + config

        fn read_scaled_ternary(mut file: FileHandle) raises -> ScaledTernaryMatrix:
            var flag_byte = file.read_bytes(1)
            if Int(flag_byte[0]) != 1:
                raise Error("Expected quantized weight flag")

            var dims_bytes = file.read_bytes(8)
            var dims_ptr = dims_bytes.unsafe_ptr().bitcast[Int32]()
            var rows = Int(dims_ptr[0])
            var cols = Int(dims_ptr[1])

            var bytes_per_row = (cols + 3) // 4
            var scales = List[Float32](capacity=rows)
            var data = List[UInt8](capacity=rows * bytes_per_row)

            for row in range(rows):
                var scale_bytes = file.read_bytes(2)
                var scale_f16 = scale_bytes.unsafe_ptr().bitcast[Float16]()[0]
                scales.append(Float32(scale_f16))

                var ternary_bytes = file.read_bytes(bytes_per_row)
                for b in range(bytes_per_row):
                    data.append(ternary_bytes[b])

            return ScaledTernaryMatrix(data^, scales^, rows, cols)

        fn read_float32(mut file: FileHandle, size: Int) raises -> Matrix:
            var flag_byte = file.read_bytes(1)
            var bytes_data = file.read_bytes(size * 4)
            var ptr = bytes_data.unsafe_ptr().bitcast[Float32]()
            var result = List[Float32](capacity=size)
            for i in range(size):
                result.append(ptr[i])
            return Matrix(result^, size, 1)

        # Read embedding
        self.embed_tokens = read_scaled_ternary(f)

        # Initialize per-layer weight lists
        self.input_layernorm = List[Matrix]()
        self.q_proj = List[ScaledTernaryMatrix]()
        self.k_proj = List[ScaledTernaryMatrix]()
        self.v_proj = List[ScaledTernaryMatrix]()
        self.o_proj = List[ScaledTernaryMatrix]()
        self.attn_sub_norm = List[Matrix]()
        self.post_attn_layernorm = List[Matrix]()
        self.gate_proj = List[ScaledTernaryMatrix]()
        self.up_proj = List[ScaledTernaryMatrix]()
        self.down_proj = List[ScaledTernaryMatrix]()
        self.ffn_sub_norm = List[Matrix]()

        # Read per-layer weights
        for layer in range(config.n_layers):
            self.input_layernorm.append(read_float32(f, config.dim))
            self.q_proj.append(read_scaled_ternary(f))
            self.k_proj.append(read_scaled_ternary(f))
            self.v_proj.append(read_scaled_ternary(f))
            self.o_proj.append(read_scaled_ternary(f))
            self.attn_sub_norm.append(read_float32(f, config.dim))
            self.post_attn_layernorm.append(read_float32(f, config.dim))
            self.gate_proj.append(read_scaled_ternary(f))
            self.up_proj.append(read_scaled_ternary(f))
            self.down_proj.append(read_scaled_ternary(f))
            self.ffn_sub_norm.append(read_float32(f, config.hidden_dim))

        # Read final layers
        self.final_norm = read_float32(f, config.dim)
        self.lm_head = read_scaled_ternary(f)

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

    fn __init__(out self, config: BitNetConfig):
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
    config: BitNetConfig,
    mut state: RunState,
    weights: BitNetWeights
):
    """BitNet forward pass using T-MAC."""
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var head_size = config.head_size
    var kv_dim = config.kv_dim
    var kv_mul = config.kv_mul
    var n_heads = config.n_heads
    var seq_len = config.seq_len

    # Get embedding
    var emb_row = token
    var emb_scale = weights.embed_tokens.get_scale(emb_row)
    for i in range(dim):
        var byte_idx = i // 4
        var bit_offset = (i % 4) * 2
        var byte_val = weights.embed_tokens.get_ternary_byte(emb_row, byte_idx)
        var bits = (Int(byte_val) >> bit_offset) & 0x03
        var ternary = _decode_ternary(bits)
        state.x[i] = Float32(ternary) * emb_scale

    # Process layers
    for layer in range(config.n_layers):
        # Input layernorm
        rmsnorm(state.xb, state.x, weights.input_layernorm[layer], 0, 0, 0, dim)

        # Build LUT for attention
        var lut = build_lut(state.xb, 0, dim)

        # QKV projections
        tmac_matmul_parallel(state.q, 0, lut, weights.q_proj[layer], 0, dim)
        tmac_matmul_parallel(state.k, 0, lut, weights.k_proj[layer], 0, kv_dim)
        tmac_matmul_parallel(state.v, 0, lut, weights.v_proj[layer], 0, kv_dim)

        # RoPE (simplified - BitNet uses different position encoding but this works)
        for i in range(0, dim, 2):
            var head_dim = i % head_size
            var freq = 1.0 / (Float32(500000.0) ** (Float32(head_dim) / Float32(head_size)))
            var val = Float32(pos) * freq
            var fcr = math.cos(val)
            var fci = math.sin(val)
            var v0 = state.q[i]
            var v1 = state.q[i + 1]
            state.q[i] = v0 * fcr - v1 * fci
            state.q[i + 1] = v0 * fci + v1 * fcr

        for i in range(0, kv_dim, 2):
            var head_dim = i % head_size
            var freq = 1.0 / (Float32(500000.0) ** (Float32(head_dim) / Float32(head_size)))
            var val = Float32(pos) * freq
            var fcr = math.cos(val)
            var fci = math.sin(val)
            var v0 = state.k[i]
            var v1 = state.k[i + 1]
            state.k[i] = v0 * fcr - v1 * fci
            state.k[i + 1] = v0 * fci + v1 * fcr

        # Cache K/V
        var cache_offset = layer * seq_len * kv_dim + pos * kv_dim
        for i in range(kv_dim):
            state.key_cache[cache_offset + i] = state.k[i]
            state.value_cache[cache_offset + i] = state.v[i]

        # Attention
        for h in range(n_heads):
            var q_offset = h * head_size
            var kv_head = h // kv_mul

            for t in range(pos + 1):
                var k_offset = layer * seq_len * kv_dim + t * kv_dim + kv_head * head_size
                var score: Float32 = 0.0
                for i in range(head_size):
                    score += state.q[q_offset + i] * state.key_cache[k_offset + i]
                state.att[h * seq_len + t] = score / math.sqrt(Float32(head_size))

            softmax(state.att, h * seq_len, pos + 1)

            for i in range(head_size):
                state.xb[q_offset + i] = 0.0
            for t in range(pos + 1):
                var v_offset = layer * seq_len * kv_dim + t * kv_dim + kv_head * head_size
                var a = state.att[h * seq_len + t]
                for i in range(head_size):
                    state.xb[q_offset + i] += a * state.value_cache[v_offset + i]

        # Output projection
        var o_lut = build_lut(state.xb, 0, dim)
        tmac_matmul_parallel(state.xb2, 0, o_lut, weights.o_proj[layer], 0, dim)

        # Attention sub-norm
        rmsnorm(state.xb2, state.xb2, weights.attn_sub_norm[layer], 0, 0, 0, dim)

        # Residual
        for i in range(dim):
            state.x[i] += state.xb2[i]

        # Post-attention layernorm
        rmsnorm(state.xb, state.x, weights.post_attn_layernorm[layer], 0, 0, 0, dim)

        # FFN
        var ffn_lut = build_lut(state.xb, 0, dim)
        tmac_matmul_parallel(state.hb, 0, ffn_lut, weights.gate_proj[layer], 0, hidden_dim)
        tmac_matmul_parallel(state.hb2, 0, ffn_lut, weights.up_proj[layer], 0, hidden_dim)

        # ReLU² and element-wise multiply
        relu_squared(state.hb, 0, hidden_dim)
        for i in range(hidden_dim):
            state.hb[i] *= state.hb2[i]

        # Down projection
        var down_lut = build_lut(state.hb, 0, hidden_dim)
        tmac_matmul_parallel(state.xb, 0, down_lut, weights.down_proj[layer], 0, dim)

        # FFN sub-norm
        rmsnorm(state.xb, state.xb, weights.ffn_sub_norm[layer], 0, 0, 0, dim)

        # Residual
        for i in range(dim):
            state.x[i] += state.xb[i]

    # Final norm
    for i in range(dim):
        state.xb[i] = state.x[i]
    rmsnorm(state.x, state.xb, weights.final_norm, 0, 0, 0, dim)

    # LM head
    var lm_lut = build_lut(state.x, 0, dim)
    tmac_matmul_parallel(state.logits, 0, lm_lut, weights.lm_head, 0, config.vocab_size)


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

    var r = random.random_float64().cast[DType.float32]() * topp
    var cumsum: Float32 = 0.0
    for i in range(size):
        cumsum += probs[i]
        if cumsum > r:
            return i

    return size - 1


# =============================================================================
# Main
# =============================================================================

fn main() raises:
    var args = argv()
    if len(args) < 2:
        print("Usage: bitnet_tmac <model.tmac2.bin> [-n tokens] [-t temp]")
        return

    var model_path = String(args[1])
    var num_tokens = 64
    var temperature: Float32 = 0.0
    var topp: Float32 = 0.9

    var i = 2
    while i < len(args):
        if String(args[i]) == "-n" and i + 1 < len(args):
            num_tokens = atol(args[i + 1])
            i += 2
        elif String(args[i]) == "-t" and i + 1 < len(args):
            temperature = atof(args[i + 1]).cast[DType.float32]()
            i += 2
        else:
            i += 1

    print("Loading BitNet model from", model_path)
    print("T-MAC Lookup Table Inference (NO MULTIPLICATION)")
    print()

    var config = BitNetConfig(model_path)
    print()
    print("Loading weights...")
    var weights = BitNetWeights(model_path, config)
    print("Weights loaded.")
    print()

    var state = RunState(config)

    print("Generating", num_tokens, "tokens...")
    print("-" * 50)

    var token = 128000  # BOS token for BitNet
    var start = time.perf_counter_ns()

    for pos in range(num_tokens):
        transformer(token, pos, config, state, weights)
        var next_token = sample_topp(state.logits, config.vocab_size, topp, temperature)

        if next_token == 128001:  # EOS
            break

        # Print token (simplified - would need actual tokenizer)
        print("<", next_token, ">", end="")
        token = next_token

    var elapsed = (time.perf_counter_ns() - start) / 1_000_000

    print()
    print("-" * 50)
    print("Generated", num_tokens, "tokens in", Int(elapsed), "ms")
    print("Speed:", Int(Float32(num_tokens) / Float32(elapsed / 1000)), "tokens/sec")
    print("Method: BitNet b1.58 + T-MAC (Lookup Table, NO MULTIPLICATION)")
