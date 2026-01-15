"""
Quantized LLaMA 2 Inference in Pure Mojo
Q8_0 quantization: 3.76x smaller, faster memory bandwidth.

Q8_0 format per block (32 weights):
- 1 float16 scale (2 bytes)
- 32 int8 values (32 bytes)
- Total: 34 bytes (vs 128 bytes float32)
"""
from collections import List, Dict
from sys import argv
from sys.info import simd_width_of
import math
import random
import time

# Constants
comptime NUM_CONFIG_INT: Int = 7
comptime BLOCK_SIZE: Int = 32
comptime BLOCK_BYTES: Int = 34  # 2 (scale) + 32 (int8s)


struct Matrix:
    """Float32 matrix for activations."""
    var data: List[Float32]
    var rows: Int
    var cols: Int

    fn __init__(out self, size: Int):
        self.data = List[Float32](capacity=size)
        for _ in range(size):
            self.data.append(0.0)
        self.rows = size
        self.cols = 1

    fn __init__(out self, rows: Int, cols: Int):
        var size = rows * cols
        self.data = List[Float32](capacity=size)
        for _ in range(size):
            self.data.append(0.0)
        self.rows = rows
        self.cols = cols

    fn __init__(out self, var data: List[Float32], rows: Int, cols: Int = 1):
        self.data = data^
        self.rows = rows
        self.cols = cols

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
    """Q8_0 quantized matrix for weights."""
    var data: List[UInt8]  # Raw bytes: [scale_f16, int8 x 32] per block
    var n_elements: Int
    var n_blocks: Int
    var rows: Int
    var cols: Int

    fn __init__(out self, var data: List[UInt8], n_elements: Int, rows: Int, cols: Int = 1):
        self.data = data^
        self.n_elements = n_elements
        self.n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        self.rows = rows
        self.cols = cols

    @always_inline
    fn get_block_scale(self, block_idx: Int) -> Float32:
        """Get scale for a block (stored as float16)."""
        var offset = block_idx * BLOCK_BYTES
        # Read float16 bytes and convert to float32
        var b0 = self.data[offset]
        var b1 = self.data[offset + 1]
        var f16_bits = (Int(b1) << 8) | Int(b0)
        return self._f16_to_f32(f16_bits)

    @always_inline
    fn get_block_value(self, block_idx: Int, idx_in_block: Int) -> Int8:
        """Get int8 value from block."""
        var offset = block_idx * BLOCK_BYTES + 2 + idx_in_block
        return Int8(self.data[offset]) if self.data[offset] < 128 else Int8(Int(self.data[offset]) - 256)

    @always_inline
    fn dequantize(self, idx: Int) -> Float32:
        """Dequantize single element."""
        var block_idx = idx // BLOCK_SIZE
        var idx_in_block = idx % BLOCK_SIZE
        var scale = self.get_block_scale(block_idx)
        var qval = self.get_block_value(block_idx, idx_in_block)
        return Float32(qval) * scale

    @staticmethod
    @always_inline
    fn _pow2(exp: Int) -> Float32:
        """Compute 2^exp for small exponents."""
        if exp >= 0:
            return Float32(1 << exp) if exp < 30 else Float32(1 << 30) * Float32(1 << (exp - 30))
        else:
            var neg_exp = -exp
            return Float32(1.0) / Float32(1 << neg_exp) if neg_exp < 30 else Float32(0.0)

    @staticmethod
    @always_inline
    fn _f16_to_f32(bits: Int) -> Float32:
        """Convert float16 bits to float32."""
        var sign = (bits >> 15) & 1
        var exp = (bits >> 10) & 0x1F
        var mant = bits & 0x3FF

        if exp == 0:
            if mant == 0:
                return Float32(-0.0) if sign else Float32(0.0)
            # Denormalized
            var val = Float32(mant) / Float32(1024.0) * QuantizedMatrix._pow2(-14)
            return -val if sign else val
        elif exp == 31:
            return Float32.MAX if mant == 0 else Float32.MAX  # Inf/NaN -> MAX
        else:
            var val = (Float32(1.0) + Float32(mant) / Float32(1024.0)) * QuantizedMatrix._pow2(exp - 15)
            return -val if sign else val


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
    var block_size: Int

    fn __init__(out self, path: String, print_config: Bool = False) raises:
        var f = open(path, "r")

        # Check magic number
        var magic = f.read_bytes(4)
        if magic[0] != ord('Q') or magic[1] != ord('8') or magic[2] != ord('V') or magic[3] != ord('1'):
            raise Error("Invalid Q8 file format")

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
        self.block_size = BLOCK_SIZE

        if not self.shared_weights:
            self.vocab_size = -self.vocab_size

        if print_config:
            print("Config: dim=", self.dim, "hidden_dim=", self.hidden_dim)
            print("        n_layers=", self.n_layers, "n_heads=", self.n_heads)
            print("        vocab_size=", self.vocab_size, "seq_len=", self.seq_len)
            print("        Q8 quantized (3.76x smaller)")


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
        var max_len_bytes = f.read_bytes(4)

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
    # Quantized weights (large matrices)
    var token_embedding: QuantizedMatrix
    var wq: QuantizedMatrix
    var wk: QuantizedMatrix
    var wv: QuantizedMatrix
    var wo: QuantizedMatrix
    var w1: QuantizedMatrix
    var w2: QuantizedMatrix
    var w3: QuantizedMatrix

    # Float32 weights (small, not quantized)
    var rms_att_weight: Matrix
    var rms_ffn_weight: Matrix
    var rms_final_weight: Matrix
    var freq_cis_real: Matrix
    var freq_cis_imag: Matrix

    fn __init__(out self, path: String, config: Config) raises:
        var f = open(path, "r")

        # Skip magic + config + block_size
        _ = f.read_bytes(4 + NUM_CONFIG_INT * 4 + 4)

        fn read_quantized(mut file: FileHandle, n_elements: Int) raises -> List[UInt8]:
            var flag = file.read_bytes(1)
            var is_quantized = flag[0] == 1
            var n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            var n_bytes = n_blocks * BLOCK_BYTES if is_quantized else n_elements * 4
            return file.read_bytes(n_bytes)

        fn read_float32(mut file: FileHandle, size: Int) raises -> List[Float32]:
            var flag = file.read_bytes(1)  # Skip flag (always 0 for float32)
            var bytes = file.read_bytes(size * 4)
            var ptr = bytes.unsafe_ptr().bitcast[Float32]()
            var result = List[Float32](capacity=size)
            for i in range(size):
                result.append(ptr[i])
            return result^

        var dim = config.dim
        var hidden_dim = config.hidden_dim
        var n_layers = config.n_layers
        var kv_dim = config.kv_dim
        var head_size = config.head_size
        var vocab_size = config.vocab_size
        var seq_len = config.seq_len

        # Load quantized weights
        self.token_embedding = QuantizedMatrix(
            read_quantized(f, vocab_size * dim),
            vocab_size * dim, vocab_size, dim
        )

        self.rms_att_weight = Matrix(
            read_float32(f, n_layers * dim), n_layers, dim
        )

        self.wq = QuantizedMatrix(
            read_quantized(f, n_layers * dim * dim),
            n_layers * dim * dim, n_layers * dim, dim
        )
        self.wk = QuantizedMatrix(
            read_quantized(f, n_layers * kv_dim * dim),
            n_layers * kv_dim * dim, n_layers * kv_dim, dim
        )
        self.wv = QuantizedMatrix(
            read_quantized(f, n_layers * kv_dim * dim),
            n_layers * kv_dim * dim, n_layers * kv_dim, dim
        )
        self.wo = QuantizedMatrix(
            read_quantized(f, n_layers * dim * dim),
            n_layers * dim * dim, n_layers * dim, dim
        )

        self.rms_ffn_weight = Matrix(
            read_float32(f, n_layers * dim), n_layers, dim
        )

        self.w1 = QuantizedMatrix(
            read_quantized(f, n_layers * hidden_dim * dim),
            n_layers * hidden_dim * dim, n_layers * hidden_dim, dim
        )
        self.w2 = QuantizedMatrix(
            read_quantized(f, n_layers * dim * hidden_dim),
            n_layers * dim * hidden_dim, n_layers * dim, hidden_dim
        )
        self.w3 = QuantizedMatrix(
            read_quantized(f, n_layers * hidden_dim * dim),
            n_layers * hidden_dim * dim, n_layers * hidden_dim, dim
        )

        self.rms_final_weight = Matrix(
            read_float32(f, dim), dim
        )
        self.freq_cis_real = Matrix(
            read_float32(f, seq_len * head_size // 2), seq_len, head_size // 2
        )
        self.freq_cis_imag = Matrix(
            read_float32(f, seq_len * head_size // 2), seq_len, head_size // 2
        )

        f.close()
        print("Loaded Q8 model:", dim * n_layers // 1000, "K parameters")


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
        self.key_cache = Matrix(config.n_layers * config.seq_len, config.kv_dim)
        self.value_cache = Matrix(config.n_layers * config.seq_len, config.kv_dim)


# =============================================================================
# Quantized operations
# =============================================================================

fn rmsnorm(mut out_mat: Matrix, out_offset: Int,
           x_mat: Matrix, x_offset: Int,
           w_mat: Matrix, w_offset: Int, size: Int):
    """RMS normalization (float32)."""
    var ss: Float32 = 0.0
    for i in range(size):
        var v = x_mat[x_offset + i]
        ss += v * v
    ss = 1.0 / math.sqrt(ss / size + 1e-5)
    for i in range(size):
        out_mat[out_offset + i] = w_mat[w_offset + i] * x_mat[x_offset + i] * ss


fn softmax(mut mat: Matrix, offset: Int, size: Int):
    """In-place softmax."""
    var max_val = mat[offset]
    for i in range(1, size):
        if mat[offset + i] > max_val:
            max_val = mat[offset + i]

    var sum_exp: Float32 = 0.0
    for i in range(size):
        var v = math.exp(mat[offset + i] - max_val)
        mat[offset + i] = v
        sum_exp += v

    var inv_sum = 1.0 / sum_exp
    for i in range(size):
        mat[offset + i] *= inv_sum


fn matmul_q8(mut out_mat: Matrix, out_offset: Int,
             x_mat: Matrix, x_offset: Int,
             w: QuantizedMatrix, w_row_offset: Int, rows: Int, cols: Int):
    """Optimized quantized matrix-vector multiplication."""
    for i in range(rows):
        var sum: Float32 = 0.0
        var row_start = (w_row_offset + i) * cols

        # Process aligned blocks (most common case)
        var n_full_blocks = cols // BLOCK_SIZE
        var block_base = row_start // BLOCK_SIZE

        for b in range(n_full_blocks):
            var block_idx = block_base + b
            var block_offset = block_idx * BLOCK_BYTES
            var x_base = b * BLOCK_SIZE

            # Read scale once per block
            var b0 = w.data[block_offset]
            var b1 = w.data[block_offset + 1]
            var f16_bits = (Int(b1) << 8) | Int(b0)
            var scale = QuantizedMatrix._f16_to_f32(f16_bits)

            # Accumulate 32 elements with single scale
            var block_sum: Float32 = 0.0
            for k in range(BLOCK_SIZE):
                var qbyte = w.data[block_offset + 2 + k]
                var qval = Int(qbyte) if qbyte < 128 else Int(qbyte) - 256
                block_sum += x_mat[x_offset + x_base + k] * Float32(qval)
            sum += block_sum * scale

        # Handle remainder
        var remainder_start = n_full_blocks * BLOCK_SIZE
        for j in range(remainder_start, cols):
            var global_idx = row_start + j
            var block_idx = global_idx // BLOCK_SIZE
            var idx_in_block = global_idx % BLOCK_SIZE
            var scale = w.get_block_scale(block_idx)
            var qval = w.get_block_value(block_idx, idx_in_block)
            sum += x_mat[x_offset + j] * Float32(qval) * scale

        out_mat[out_offset + i] = sum


fn get_embedding_q8(mut out_mat: Matrix, token: Int, w: QuantizedMatrix, dim: Int):
    """Get token embedding from quantized matrix."""
    var start = token * dim
    for i in range(dim):
        out_mat[i] = w.dequantize(start + i)


fn transformer_forward(
    token: Int,
    pos: Int,
    config: Config,
    mut state: RunState,
    weights: TransformerWeights,
) raises:
    """Forward pass with quantized weights."""
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var head_size = config.head_size
    var kv_dim = config.kv_dim
    var kv_mul = config.kv_mul
    var n_heads = config.n_heads
    var n_kv_heads = config.n_kv_heads
    var sqrt_head_size = math.sqrt(Float32(head_size))

    # Get token embedding (dequantize)
    get_embedding_q8(state.x, token, weights.token_embedding, dim)

    var freq_offset = pos * (head_size // 2)

    for layer in range(config.n_layers):
        var layer_dim_offset = layer * dim

        # Attention rmsnorm
        rmsnorm(state.xb, 0, state.x, 0,
                weights.rms_att_weight, layer_dim_offset, dim)

        # QKV projections (quantized)
        var wq_row_offset = layer * dim
        var wk_row_offset = layer * kv_dim
        var wv_row_offset = layer * kv_dim

        matmul_q8(state.q, 0, state.xb, 0, weights.wq, wq_row_offset, dim, dim)

        var cache_offset = layer * config.seq_len * kv_dim + pos * kv_dim
        matmul_q8(state.key_cache, cache_offset, state.xb, 0, weights.wk, wk_row_offset, kv_dim, dim)
        matmul_q8(state.value_cache, cache_offset, state.xb, 0, weights.wv, wv_row_offset, kv_dim, dim)

        # RoPE
        for h in range(n_heads):
            for j in range(0, head_size, 2):
                var fcr = weights.freq_cis_real[freq_offset + j // 2]
                var fci = weights.freq_cis_imag[freq_offset + j // 2]

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

        # Attention
        for h in range(n_heads):
            var q_offset = h * head_size
            var att_offset = h * config.seq_len

            for t in range(pos + 1):
                var k_base = layer * config.seq_len * kv_dim + t * kv_dim + (h // kv_mul) * head_size
                var score: Float32 = 0.0
                for i in range(head_size):
                    score += state.q[q_offset + i] * state.key_cache[k_base + i]
                state.att[att_offset + t] = score / sqrt_head_size

            softmax(state.att, att_offset, pos + 1)

            var xb_offset = h * head_size
            for t in range(pos + 1):
                var v_base = layer * config.seq_len * kv_dim + t * kv_dim + (h // kv_mul) * head_size
                var a = state.att[att_offset + t]
                for i in range(head_size):
                    state.xb[xb_offset + i] += a * state.value_cache[v_base + i]

        # Output projection (quantized)
        var wo_row_offset = layer * dim
        matmul_q8(state.xb2, 0, state.xb, 0, weights.wo, wo_row_offset, dim, dim)

        for i in range(dim):
            state.x[i] += state.xb2[i]

        # FFN rmsnorm
        rmsnorm(state.xb, 0, state.x, 0,
                weights.rms_ffn_weight, layer_dim_offset, dim)

        # FFN (quantized)
        var w1_row_offset = layer * hidden_dim
        var w3_row_offset = layer * hidden_dim
        matmul_q8(state.hb, 0, state.xb, 0, weights.w1, w1_row_offset, hidden_dim, dim)
        matmul_q8(state.hb2, 0, state.xb, 0, weights.w3, w3_row_offset, hidden_dim, dim)

        # SiLU
        for i in range(hidden_dim):
            var v = state.hb[i]
            state.hb[i] = v * (1.0 / (1.0 + math.exp(-v))) * state.hb2[i]

        var w2_row_offset = layer * dim
        matmul_q8(state.xb, 0, state.hb, 0, weights.w2, w2_row_offset, dim, hidden_dim)

        for i in range(dim):
            state.x[i] += state.xb[i]

    # Final rmsnorm
    rmsnorm(state.xb, 0, state.x, 0, weights.rms_final_weight, 0, dim)

    # Classifier (quantized)
    matmul_q8(state.logits, 0, state.xb, 0, weights.token_embedding, 0, config.vocab_size, dim)


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
    elif s == "<0x09>":
        print("\t", end="")
    else:
        print(s, end="")


fn main() raises:
    var checkpoint = "stories110M.q8.bin"
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

    print("Loading Q8 model from", checkpoint)
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
                softmax(state.logits, 0, config.vocab_size)
                next_token = sample(state.logits, config.vocab_size)

        if next_token == 1 or next_token == 2:
            break

        print_token(tokenizer, next_token)
        token = next_token
        tokens_generated += 1

    var end_time = time.perf_counter_ns()
    var elapsed_ms = (end_time - start_time) // 1_000_000

    print("\n" + "-" * 50)
    print("Generated", tokens_generated, "tokens in", elapsed_ms, "ms")
    if elapsed_ms > 0:
        var tok_per_sec = tokens_generated * 1000 // Int(elapsed_ms)
        print("Speed:", tok_per_sec, "tokens/sec")
