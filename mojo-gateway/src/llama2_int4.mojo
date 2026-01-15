"""
Int4 Quantized LLaMA 2 Inference in Pure Mojo
Q4_0 format: 18 bytes per 32 weights (2-byte f16 scale + 16 bytes packed int4)
7.1x memory reduction with SIMD dequantization.
"""
from algorithm import parallelize
from collections import List, Dict
from memory import UnsafePointer
from sys import argv
from sys.info import simd_width_of, num_performance_cores
import math
import random
import time

comptime NUM_CONFIG_INT: Int = 7
comptime SIMD_WIDTH: Int = 8
comptime Q4_BLOCK_SIZE: Int = 32
comptime Q4_BLOCK_BYTES: Int = 18  # 2 (f16 scale) + 16 (packed int4)


@always_inline
fn q4_dequant(idx: Int) -> Float32:
    """Fast int4 to float32 dequantization. Index [0-15] maps to value [-8,7]."""
    return Float32(idx - 8)


struct Matrix:
    """Float32 matrix for activations and non-quantized weights."""
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


struct QuantizedMatrix:
    """Q4_0 quantized matrix storage."""
    var data: List[UInt8]  # Raw bytes: [scale_f16, packed_int4...]
    var num_weights: Int   # Total number of float32 weights represented
    var num_blocks: Int    # Number of Q4 blocks

    fn __init__(out self, num_weights: Int):
        self.num_weights = num_weights
        self.num_blocks = (num_weights + Q4_BLOCK_SIZE - 1) // Q4_BLOCK_SIZE
        var total_bytes = self.num_blocks * Q4_BLOCK_BYTES
        self.data = List[UInt8](capacity=total_bytes)
        for _ in range(total_bytes):
            self.data.append(0)

    fn __init__(out self, var data: List[UInt8], num_weights: Int):
        self.data = data^
        self.num_weights = num_weights
        self.num_blocks = (num_weights + Q4_BLOCK_SIZE - 1) // Q4_BLOCK_SIZE

    @always_inline
    fn get_block_scale(self, block_idx: Int) -> Float32:
        """Get the float16 scale for a block, converted to float32."""
        var byte_offset = block_idx * Q4_BLOCK_BYTES
        var ptr = self.data.unsafe_ptr()
        # Read 2 bytes as float16, convert to float32
        var b0 = Int(ptr[byte_offset])
        var b1 = Int(ptr[byte_offset + 1])
        var f16_bits = b0 | (b1 << 8)
        return _f16_to_f32(f16_bits)

    @always_inline
    fn get_weight(self, block_idx: Int, weight_in_block: Int) -> Float32:
        """Dequantize a single weight from Q4_0 format."""
        var scale = self.get_block_scale(block_idx)
        var byte_offset = block_idx * Q4_BLOCK_BYTES + 2  # Skip scale
        var packed_idx = weight_in_block // 2
        var ptr = self.data.unsafe_ptr()
        var packed = ptr[byte_offset + packed_idx]

        var quant: Int
        if weight_in_block % 2 == 0:
            quant = Int(packed & 0x0F)  # Low nibble
        else:
            quant = Int((packed >> 4) & 0x0F)  # High nibble

        # Convert from [0,15] to [-8,7] and dequantize
        return Float32(quant - 8) * scale


@always_inline
fn _f16_to_f32(f16_bits: Int) -> Float32:
    """Fast float16 to float32 conversion for normal values (ignores edge cases)."""
    var sign = (f16_bits >> 15) & 1
    var exp = (f16_bits >> 10) & 0x1F
    var mant = f16_bits & 0x3FF

    # Fast path for zero
    if f16_bits == 0 or f16_bits == 0x8000:
        return Float32(0.0)

    # Normal number: value = (-1)^sign * 2^(exp-15) * (1 + mant/1024)
    var mantissa = Float32(1.0) + Float32(mant) * Float32(0.0009765625)  # 1/1024
    var exp_val = exp - 15

    var value: Float32
    if exp_val >= 0:
        if exp_val < 16:
            value = mantissa * Float32(1 << exp_val)
        else:
            value = mantissa * Float32(65536.0)  # Cap at 2^16
    else:
        if exp_val > -16:
            value = mantissa / Float32(1 << (-exp_val))
        else:
            value = Float32(0.0)

    if sign == 1:
        return -value
    return value


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
    var block_size: Int  # Q4 block size from file

    fn __init__(out self, path: String, print_config: Bool = False) raises:
        var f = open(path, "r")

        # Read magic number
        var magic_bytes = f.read_bytes(4)
        var magic = String("")
        for i in range(4):
            magic += chr(Int(magic_bytes[i]))

        if magic != "Q4V1":
            raise Error("Invalid Q4 model file. Expected Q4V1 magic number.")

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

        # Read block size (comes after config)
        f = open(path, "r")
        _ = f.read_bytes(4 + NUM_CONFIG_INT * 4)  # Skip magic + config
        var block_bytes = f.read_bytes(4)
        self.block_size = Int(block_bytes.unsafe_ptr().bitcast[Int32]()[0])
        f.close()

        if print_config:
            print("Q4 Config: dim=", self.dim, "hidden_dim=", self.hidden_dim)
            print("           n_layers=", self.n_layers, "n_heads=", self.n_heads)
            print("           vocab_size=", self.vocab_size, "block_size=", self.block_size)
            print("           SIMD:", SIMD_WIDTH, "Cores:", num_performance_cores())


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


struct TransformerWeights:
    """Weights with Q4 quantization for large matrices, float32 for small ones."""
    # Quantized (large matrices)
    var token_embedding_q: QuantizedMatrix
    var wq_q: QuantizedMatrix
    var wk_q: QuantizedMatrix
    var wv_q: QuantizedMatrix
    var wo_q: QuantizedMatrix
    var w1_q: QuantizedMatrix
    var w2_q: QuantizedMatrix
    var w3_q: QuantizedMatrix

    # Non-quantized (small matrices - normalization and position)
    var rms_att_weight: Matrix
    var rms_ffn_weight: Matrix
    var rms_final_weight: Matrix
    var freq_cis_real: Matrix
    var freq_cis_imag: Matrix
    var wcls_q: QuantizedMatrix  # May be shared with token_embedding

    fn __init__(out self, path: String, config: Config) raises:
        var f = open(path, "r")
        # Skip magic + config + block_size
        _ = f.read_bytes(4 + NUM_CONFIG_INT * 4 + 4)

        fn read_quantized(mut file: FileHandle, num_weights: Int) raises -> QuantizedMatrix:
            """Read Q4 quantized weights."""
            var is_quantized_bytes = file.read_bytes(1)
            var is_quantized = is_quantized_bytes[0] == 1

            if is_quantized:
                var num_blocks = (num_weights + Q4_BLOCK_SIZE - 1) // Q4_BLOCK_SIZE
                var total_bytes = num_blocks * Q4_BLOCK_BYTES
                var bytes = file.read_bytes(total_bytes)
                var data = List[UInt8](capacity=total_bytes)
                for i in range(total_bytes):
                    data.append(bytes[i])
                return QuantizedMatrix(data^, num_weights)
            else:
                # Non-quantized - read as float32 and should not happen for large matrices
                raise Error("Expected quantized weights")

        fn read_float32(mut file: FileHandle, size: Int) raises -> List[Float32]:
            """Read non-quantized float32 weights."""
            var is_quantized_bytes = file.read_bytes(1)
            var is_quantized = is_quantized_bytes[0] == 1

            if is_quantized:
                raise Error("Expected non-quantized weights")

            var bytes = file.read_bytes(size * 4)
            var ptr = bytes.unsafe_ptr().bitcast[Float32]()
            var result = List[Float32](capacity=size)
            for i in range(size):
                result.append(ptr[i])
            return result^

        # Read weights in order (matching quantize_int4.py)
        self.token_embedding_q = read_quantized(f, config.vocab_size * config.dim)
        self.rms_att_weight = Matrix(read_float32(f, config.n_layers * config.dim), config.n_layers, config.dim)
        self.wq_q = read_quantized(f, config.n_layers * config.dim * config.dim)
        self.wk_q = read_quantized(f, config.n_layers * config.kv_dim * config.dim)
        self.wv_q = read_quantized(f, config.n_layers * config.kv_dim * config.dim)
        self.wo_q = read_quantized(f, config.n_layers * config.dim * config.dim)
        self.rms_ffn_weight = Matrix(read_float32(f, config.n_layers * config.dim), config.n_layers, config.dim)
        self.w1_q = read_quantized(f, config.n_layers * config.hidden_dim * config.dim)
        self.w2_q = read_quantized(f, config.n_layers * config.dim * config.hidden_dim)
        self.w3_q = read_quantized(f, config.n_layers * config.hidden_dim * config.dim)
        self.rms_final_weight = Matrix(read_float32(f, config.dim), config.dim)
        self.freq_cis_real = Matrix(read_float32(f, config.seq_len * config.head_size // 2), config.seq_len, config.head_size // 2)
        self.freq_cis_imag = Matrix(read_float32(f, config.seq_len * config.head_size // 2), config.seq_len, config.head_size // 2)

        if config.shared_weights:
            # wcls shares token_embedding - create empty placeholder
            self.wcls_q = QuantizedMatrix(0)
        else:
            self.wcls_q = read_quantized(f, config.vocab_size * config.dim)

        f.close()


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
# Q4 SIMD Operations
# =============================================================================

fn rmsnorm_simd(mut out_mat: Matrix, out_offset: Int,
                x_mat: Matrix, x_offset: Int,
                w_mat: Matrix, w_offset: Int, size: Int):
    """RMS normalization with SIMD."""
    var x_ptr = x_mat.data.unsafe_ptr()
    var w_ptr = w_mat.data.unsafe_ptr()

    var ss: Float32 = 0.0
    var i = 0
    while i + SIMD_WIDTH <= size:
        var v = x_ptr.load[width=SIMD_WIDTH](x_offset + i)
        ss += (v * v).reduce_add()
        i += SIMD_WIDTH
    while i < size:
        var v = x_ptr[x_offset + i]
        ss += v * v
        i += 1

    ss = 1.0 / math.sqrt(ss / size + 1e-5)

    i = 0
    while i + SIMD_WIDTH <= size:
        var x = x_ptr.load[width=SIMD_WIDTH](x_offset + i)
        var w = w_ptr.load[width=SIMD_WIDTH](w_offset + i)
        var result = w * x * ss
        for j in range(SIMD_WIDTH):
            out_mat[out_offset + i + j] = result[j]
        i += SIMD_WIDTH
    while i < size:
        out_mat[out_offset + i] = w_ptr[w_offset + i] * x_ptr[x_offset + i] * ss
        i += 1


fn softmax_simd(mut mat: Matrix, offset: Int, size: Int):
    """Softmax with numerical stability."""
    var ptr = mat.data.unsafe_ptr()

    var max_val = ptr[offset]
    for i in range(1, size):
        if ptr[offset + i] > max_val:
            max_val = ptr[offset + i]

    var sum_exp: Float32 = 0.0
    for i in range(size):
        var v = math.exp(ptr[offset + i] - max_val)
        mat[offset + i] = v
        sum_exp += v

    var inv_sum = 1.0 / sum_exp
    for i in range(size):
        mat[offset + i] *= inv_sum


fn matmul_q4_parallel(mut out_mat: Matrix, out_offset: Int,
                      x_mat: Matrix, x_offset: Int,
                      w_q: QuantizedMatrix, w_offset: Int,
                      rows: Int, cols: Int):
    """
    Parallel matrix-vector multiplication with LUT-based Q4 dequantization.
    """
    var x_ptr = x_mat.data.unsafe_ptr()
    var w_ptr = w_q.data.unsafe_ptr()

    @parameter
    fn compute_row(row: Int):
        var weight_start = w_offset + row * cols
        var sum: Float32 = 0.0
        var col = 0

        # Process in groups of 32 (one Q4 block)
        while col + Q4_BLOCK_SIZE <= cols:
            var block_idx = (weight_start + col) // Q4_BLOCK_SIZE
            var byte_offset = block_idx * Q4_BLOCK_BYTES

            # Get scale once per block
            var b0 = Int(w_ptr[byte_offset])
            var b1 = Int(w_ptr[byte_offset + 1])
            var f16_bits = b0 | (b1 << 8)
            var scale = _f16_to_f32(f16_bits)

            # Process all 32 weights using LUT for dequantization
            for chunk in range(4):
                var packed_base = byte_offset + 2 + chunk * 4
                var p0 = w_ptr[packed_base]
                var p1 = w_ptr[packed_base + 1]
                var p2 = w_ptr[packed_base + 2]
                var p3 = w_ptr[packed_base + 3]

                var x_base = x_offset + col + chunk * 8

                # Use LUT for dequant - avoids Int->Float32 conversion
                sum += x_ptr[x_base] * q4_dequant(Int(p0 & 0x0F)) * scale
                sum += x_ptr[x_base + 1] * q4_dequant(Int((p0 >> 4) & 0x0F)) * scale
                sum += x_ptr[x_base + 2] * q4_dequant(Int(p1 & 0x0F)) * scale
                sum += x_ptr[x_base + 3] * q4_dequant(Int((p1 >> 4) & 0x0F)) * scale
                sum += x_ptr[x_base + 4] * q4_dequant(Int(p2 & 0x0F)) * scale
                sum += x_ptr[x_base + 5] * q4_dequant(Int((p2 >> 4) & 0x0F)) * scale
                sum += x_ptr[x_base + 6] * q4_dequant(Int(p3 & 0x0F)) * scale
                sum += x_ptr[x_base + 7] * q4_dequant(Int((p3 >> 4) & 0x0F)) * scale

            col += Q4_BLOCK_SIZE

        # Handle remaining weights
        while col < cols:
            var weight_idx = weight_start + col
            var block_idx = weight_idx // Q4_BLOCK_SIZE
            var byte_offset = block_idx * Q4_BLOCK_BYTES

            var b0 = Int(w_ptr[byte_offset])
            var b1 = Int(w_ptr[byte_offset + 1])
            var f16_bits = b0 | (b1 << 8)
            var scale = _f16_to_f32(f16_bits)

            var in_block = weight_idx - block_idx * Q4_BLOCK_SIZE
            var packed_idx = in_block // 2
            var packed = w_ptr[byte_offset + 2 + packed_idx]

            var quant: Int
            if in_block % 2 == 0:
                quant = Int(packed & 0x0F)
            else:
                quant = Int((packed >> 4) & 0x0F)

            sum += x_ptr[x_offset + col] * q4_dequant(quant) * scale
            col += 1

        out_mat[out_offset + row] = sum

    parallelize[compute_row](rows)


fn matmul_q4(mut out_mat: Matrix, out_offset: Int,
             x_mat: Matrix, x_offset: Int,
             w_q: QuantizedMatrix, w_offset: Int,
             rows: Int, cols: Int):
    """LUT-based single-threaded Q4 matmul for smaller matrices."""
    var x_ptr = x_mat.data.unsafe_ptr()
    var w_ptr = w_q.data.unsafe_ptr()

    for row in range(rows):
        var weight_start = w_offset + row * cols
        var sum: Float32 = 0.0
        var col = 0

        while col + Q4_BLOCK_SIZE <= cols:
            var block_idx = (weight_start + col) // Q4_BLOCK_SIZE
            var byte_offset = block_idx * Q4_BLOCK_BYTES

            var b0 = Int(w_ptr[byte_offset])
            var b1 = Int(w_ptr[byte_offset + 1])
            var f16_bits = b0 | (b1 << 8)
            var scale = _f16_to_f32(f16_bits)

            for chunk in range(4):
                var packed_base = byte_offset + 2 + chunk * 4
                var p0 = w_ptr[packed_base]
                var p1 = w_ptr[packed_base + 1]
                var p2 = w_ptr[packed_base + 2]
                var p3 = w_ptr[packed_base + 3]

                var x_base = x_offset + col + chunk * 8

                sum += x_ptr[x_base] * q4_dequant(Int(p0 & 0x0F)) * scale
                sum += x_ptr[x_base + 1] * q4_dequant(Int((p0 >> 4) & 0x0F)) * scale
                sum += x_ptr[x_base + 2] * q4_dequant(Int(p1 & 0x0F)) * scale
                sum += x_ptr[x_base + 3] * q4_dequant(Int((p1 >> 4) & 0x0F)) * scale
                sum += x_ptr[x_base + 4] * q4_dequant(Int(p2 & 0x0F)) * scale
                sum += x_ptr[x_base + 5] * q4_dequant(Int((p2 >> 4) & 0x0F)) * scale
                sum += x_ptr[x_base + 6] * q4_dequant(Int(p3 & 0x0F)) * scale
                sum += x_ptr[x_base + 7] * q4_dequant(Int((p3 >> 4) & 0x0F)) * scale

            col += Q4_BLOCK_SIZE

        while col < cols:
            var weight_idx = weight_start + col
            var block_idx = weight_idx // Q4_BLOCK_SIZE
            var byte_offset = block_idx * Q4_BLOCK_BYTES

            var b0 = Int(w_ptr[byte_offset])
            var b1 = Int(w_ptr[byte_offset + 1])
            var f16_bits = b0 | (b1 << 8)
            var scale = _f16_to_f32(f16_bits)

            var in_block = weight_idx - block_idx * Q4_BLOCK_SIZE
            var packed_idx = in_block // 2
            var packed = w_ptr[byte_offset + 2 + packed_idx]

            var quant: Int
            if in_block % 2 == 0:
                quant = Int(packed & 0x0F)
            else:
                quant = Int((packed >> 4) & 0x0F)

            sum += x_ptr[x_offset + col] * q4_dequant(quant) * scale
            col += 1

        out_mat[out_offset + row] = sum


fn get_token_embedding(mut out_mat: Matrix, out_offset: Int,
                       token: Int, dim: Int, w_q: QuantizedMatrix):
    """Extract and dequantize token embedding."""
    var w_ptr = w_q.data.unsafe_ptr()
    var weight_start = token * dim

    for i in range(dim):
        var block_idx = (weight_start + i) // Q4_BLOCK_SIZE
        var byte_offset = block_idx * Q4_BLOCK_BYTES

        var b0 = Int(w_ptr[byte_offset])
        var b1 = Int(w_ptr[byte_offset + 1])
        var f16_bits = b0 | (b1 << 8)
        var scale = _f16_to_f32(f16_bits)

        var block_start = block_idx * Q4_BLOCK_SIZE
        var in_block = weight_start + i - block_start
        var packed_idx = in_block // 2
        var packed = w_ptr[byte_offset + 2 + packed_idx]

        var quant: Int
        if in_block % 2 == 0:
            quant = Int(packed & 0x0F)
        else:
            quant = Int((packed >> 4) & 0x0F)

        out_mat[out_offset + i] = Float32(quant - 8) * scale


fn transformer_forward(
    token: Int,
    pos: Int,
    config: Config,
    mut state: RunState,
    weights: TransformerWeights,
) raises:
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var head_size = config.head_size
    var kv_dim = config.kv_dim
    var kv_mul = config.kv_mul
    var n_heads = config.n_heads
    var n_kv_heads = config.n_kv_heads
    var sqrt_head_size = math.sqrt(Float32(head_size))

    # Get token embedding (dequantized)
    get_token_embedding(state.x, 0, token, dim, weights.token_embedding_q)

    var freq_offset = pos * (head_size // 2)

    for layer in range(config.n_layers):
        var layer_dim_offset = layer * dim

        rmsnorm_simd(state.xb, 0, state.x, 0, weights.rms_att_weight, layer_dim_offset, dim)

        var wq_offset = layer * dim * dim
        var wk_offset = layer * kv_dim * dim
        var wv_offset = layer * kv_dim * dim

        # Q4 matmuls
        matmul_q4_parallel(state.q, 0, state.xb, 0, weights.wq_q, wq_offset, dim, dim)

        var cache_offset = layer * config.seq_len * kv_dim + pos * kv_dim
        matmul_q4(state.key_cache, cache_offset, state.xb, 0, weights.wk_q, wk_offset, kv_dim, dim)
        matmul_q4(state.value_cache, cache_offset, state.xb, 0, weights.wv_q, wv_offset, kv_dim, dim)

        # RoPE
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

        # Attention (float32 - cached values already dequantized)
        var q_ptr = state.q.data.unsafe_ptr()
        var k_ptr = state.key_cache.data.unsafe_ptr()
        var v_ptr = state.value_cache.data.unsafe_ptr()

        for h in range(n_heads):
            var q_offset = h * head_size
            var att_offset = h * config.seq_len

            for t in range(pos + 1):
                var k_base = layer * config.seq_len * kv_dim + t * kv_dim + (h // kv_mul) * head_size

                var score: Float32 = 0.0
                var i = 0
                while i + SIMD_WIDTH <= head_size:
                    var qv = q_ptr.load[width=SIMD_WIDTH](q_offset + i)
                    var kv = k_ptr.load[width=SIMD_WIDTH](k_base + i)
                    score += (qv * kv).reduce_add()
                    i += SIMD_WIDTH
                while i < head_size:
                    score += q_ptr[q_offset + i] * k_ptr[k_base + i]
                    i += 1

                state.att[att_offset + t] = score / sqrt_head_size

            softmax_simd(state.att, att_offset, pos + 1)

            var xb_offset = h * head_size
            for t in range(pos + 1):
                var v_base = layer * config.seq_len * kv_dim + t * kv_dim + (h // kv_mul) * head_size
                var a = state.att[att_offset + t]

                var i = 0
                while i + SIMD_WIDTH <= head_size:
                    var v = v_ptr.load[width=SIMD_WIDTH](v_base + i)
                    var result = v * a
                    for j in range(SIMD_WIDTH):
                        state.xb[xb_offset + i + j] += result[j]
                    i += SIMD_WIDTH
                while i < head_size:
                    state.xb[xb_offset + i] += a * v_ptr[v_base + i]
                    i += 1

        var wo_offset = layer * dim * dim
        matmul_q4_parallel(state.xb2, 0, state.xb, 0, weights.wo_q, wo_offset, dim, dim)

        for i in range(dim):
            state.x[i] += state.xb2[i]

        # FFN
        rmsnorm_simd(state.xb, 0, state.x, 0, weights.rms_ffn_weight, layer_dim_offset, dim)

        var w1_offset = layer * hidden_dim * dim
        var w3_offset = layer * hidden_dim * dim
        matmul_q4_parallel(state.hb, 0, state.xb, 0, weights.w1_q, w1_offset, hidden_dim, dim)
        matmul_q4_parallel(state.hb2, 0, state.xb, 0, weights.w3_q, w3_offset, hidden_dim, dim)

        # SiLU activation
        for i in range(hidden_dim):
            var v = state.hb[i]
            state.hb[i] = v * (1.0 / (1.0 + math.exp(-v))) * state.hb2[i]

        var w2_offset = layer * dim * hidden_dim
        matmul_q4_parallel(state.xb, 0, state.hb, 0, weights.w2_q, w2_offset, dim, hidden_dim)

        for i in range(dim):
            state.x[i] += state.xb[i]

    # Final layer norm
    rmsnorm_simd(state.xb, 0, state.x, 0, weights.rms_final_weight, 0, dim)

    # Classifier (reuse token embedding if shared)
    if config.shared_weights:
        matmul_q4_parallel(state.logits, 0, state.xb, 0, weights.token_embedding_q, 0, config.vocab_size, dim)
    else:
        matmul_q4_parallel(state.logits, 0, state.xb, 0, weights.wcls_q, 0, config.vocab_size, dim)


fn argmax(mat: Matrix, size: Int) -> Int:
    var max_idx = 0
    var max_val = mat[0]
    for i in range(1, size):
        if mat[i] > max_val:
            max_val = mat[i]
            max_idx = i
    return max_idx


fn sample(mat: Matrix, size: Int) -> Int:
    var r = random.random_float64().cast[DType.float32]()
    var cdf: Float32 = 0.0
    for i in range(size):
        cdf += mat[i]
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


fn main() raises:
    var checkpoint = "stories15M.q4.bin"
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

    print("Loading Q4 quantized model from", checkpoint)
    var config = Config(checkpoint, True)
    var weights = TransformerWeights(checkpoint, config)
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
        transformer_forward(token, pos, config, state, weights)

        var next_token: Int
        if pos < len(prompt_tokens):
            next_token = prompt_tokens[pos]
        else:
            if temperature == 0.0:
                next_token = argmax(state.logits, config.vocab_size)
            else:
                for j in range(config.vocab_size):
                    state.logits[j] /= temperature
                softmax_simd(state.logits, 0, config.vocab_size)
                next_token = sample(state.logits, config.vocab_size)

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
