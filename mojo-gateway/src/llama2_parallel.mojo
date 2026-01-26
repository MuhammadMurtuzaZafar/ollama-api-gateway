"""
Parallel + SIMD LLaMA 2 Inference in Pure Mojo
Multi-threaded matmul for maximum CPU utilization.
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
            print("Config: dim=", self.dim, "hidden_dim=", self.hidden_dim)
            print("        n_layers=", self.n_layers, "n_heads=", self.n_heads)
            print("        vocab_size=", self.vocab_size)
            print("        SIMD:", SIMD_WIDTH, "Cores:", num_performance_cores())


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
    var token_embedding: Matrix
    var rms_att_weight: Matrix
    var wq: Matrix
    var wk: Matrix
    var wv: Matrix
    var wo: Matrix
    var rms_ffn_weight: Matrix
    var w1: Matrix
    var w2: Matrix
    var w3: Matrix
    var rms_final_weight: Matrix
    var freq_cis_real: Matrix
    var freq_cis_imag: Matrix
    var wcls: Matrix

    fn __init__(out self, path: String, config: Config) raises:
        var f = open(path, "r")
        _ = f.read_bytes(NUM_CONFIG_INT * 4)

        fn read_weights(mut file: FileHandle, size: Int) raises -> List[Float32]:
            var bytes = file.read_bytes(size * 4)
            var ptr = bytes.unsafe_ptr().bitcast[Float32]()
            var result = List[Float32](capacity=size)
            for i in range(size):
                result.append(ptr[i])
            return result^

        self.token_embedding = Matrix(read_weights(f, config.vocab_size * config.dim), config.vocab_size, config.dim)
        self.rms_att_weight = Matrix(read_weights(f, config.n_layers * config.dim), config.n_layers, config.dim)
        self.wq = Matrix(read_weights(f, config.n_layers * config.dim * config.dim), config.n_layers, config.dim, config.dim)
        self.wk = Matrix(read_weights(f, config.n_layers * config.kv_dim * config.dim), config.n_layers, config.kv_dim, config.dim)
        self.wv = Matrix(read_weights(f, config.n_layers * config.kv_dim * config.dim), config.n_layers, config.kv_dim, config.dim)
        self.wo = Matrix(read_weights(f, config.n_layers * config.dim * config.dim), config.n_layers, config.dim, config.dim)
        self.rms_ffn_weight = Matrix(read_weights(f, config.n_layers * config.dim), config.n_layers, config.dim)
        self.w1 = Matrix(read_weights(f, config.n_layers * config.hidden_dim * config.dim), config.n_layers, config.hidden_dim, config.dim)
        self.w2 = Matrix(read_weights(f, config.n_layers * config.dim * config.hidden_dim), config.n_layers, config.dim, config.hidden_dim)
        self.w3 = Matrix(read_weights(f, config.n_layers * config.hidden_dim * config.dim), config.n_layers, config.hidden_dim, config.dim)
        self.rms_final_weight = Matrix(read_weights(f, config.dim), config.dim)
        self.freq_cis_real = Matrix(read_weights(f, config.seq_len * config.head_size // 2), config.seq_len, config.head_size // 2)
        self.freq_cis_imag = Matrix(read_weights(f, config.seq_len * config.head_size // 2), config.seq_len, config.head_size // 2)

        if config.shared_weights:
            self.wcls = Matrix(List[Float32](), 0, 0)
        else:
            self.wcls = Matrix(read_weights(f, config.vocab_size * config.dim), config.vocab_size, config.dim)

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
# Parallel + SIMD operations
# =============================================================================

fn rmsnorm_simd(mut out_mat: Matrix, out_offset: Int,
                x_mat: Matrix, x_offset: Int,
                w_mat: Matrix, w_offset: Int, size: Int):
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


fn matmul_simd_parallel(mut out_mat: Matrix, out_offset: Int,
                        x_mat: Matrix, x_offset: Int,
                        w_mat: Matrix, w_offset: Int, rows: Int, cols: Int):
    """Parallel SIMD matrix-vector multiplication."""
    var x_ptr = x_mat.data.unsafe_ptr()
    var w_ptr = w_mat.data.unsafe_ptr()

    @parameter
    fn compute_row(i: Int):
        var row_off = w_offset + i * cols
        var sum: Float32 = 0.0
        var j = 0
        while j + SIMD_WIDTH <= cols:
            var xv = x_ptr.load[width=SIMD_WIDTH](x_offset + j)
            var wv = w_ptr.load[width=SIMD_WIDTH](row_off + j)
            sum += (xv * wv).reduce_add()
            j += SIMD_WIDTH
        while j < cols:
            sum += x_ptr[x_offset + j] * w_ptr[row_off + j]
            j += 1
        out_mat[out_offset + i] = sum

    parallelize[compute_row](rows)


fn matmul_simd(mut out_mat: Matrix, out_offset: Int,
               x_mat: Matrix, x_offset: Int,
               w_mat: Matrix, w_offset: Int, rows: Int, cols: Int):
    """SIMD matrix-vector multiplication (single-threaded for small matrices)."""
    var x_ptr = x_mat.data.unsafe_ptr()
    var w_ptr = w_mat.data.unsafe_ptr()

    for i in range(rows):
        var row_off = w_offset + i * cols
        var sum: Float32 = 0.0
        var j = 0
        while j + SIMD_WIDTH <= cols:
            var xv = x_ptr.load[width=SIMD_WIDTH](x_offset + j)
            var wv = w_ptr.load[width=SIMD_WIDTH](row_off + j)
            sum += (xv * wv).reduce_add()
            j += SIMD_WIDTH
        while j < cols:
            sum += x_ptr[x_offset + j] * w_ptr[row_off + j]
            j += 1
        out_mat[out_offset + i] = sum


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

    var emb_ptr = weights.token_embedding.data.unsafe_ptr()
    var emb_offset = token * dim
    for i in range(dim):
        state.x[i] = emb_ptr[emb_offset + i]

    var freq_offset = pos * (head_size // 2)

    for layer in range(config.n_layers):
        var layer_dim_offset = layer * dim

        rmsnorm_simd(state.xb, 0, state.x, 0, weights.rms_att_weight, layer_dim_offset, dim)

        var wq_offset = layer * dim * dim
        var wk_offset = layer * kv_dim * dim
        var wv_offset = layer * kv_dim * dim

        # Use parallel matmul for larger matrices
        matmul_simd_parallel(state.q, 0, state.xb, 0, weights.wq, wq_offset, dim, dim)

        var cache_offset = layer * config.seq_len * kv_dim + pos * kv_dim
        matmul_simd(state.key_cache, cache_offset, state.xb, 0, weights.wk, wk_offset, kv_dim, dim)
        matmul_simd(state.value_cache, cache_offset, state.xb, 0, weights.wv, wv_offset, kv_dim, dim)

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
        matmul_simd_parallel(state.xb2, 0, state.xb, 0, weights.wo, wo_offset, dim, dim)

        for i in range(dim):
            state.x[i] += state.xb2[i]

        rmsnorm_simd(state.xb, 0, state.x, 0, weights.rms_ffn_weight, layer_dim_offset, dim)

        var w1_offset = layer * hidden_dim * dim
        var w3_offset = layer * hidden_dim * dim
        matmul_simd_parallel(state.hb, 0, state.xb, 0, weights.w1, w1_offset, hidden_dim, dim)
        matmul_simd_parallel(state.hb2, 0, state.xb, 0, weights.w3, w3_offset, hidden_dim, dim)

        for i in range(hidden_dim):
            var v = state.hb[i]
            state.hb[i] = v * (1.0 / (1.0 + math.exp(-v))) * state.hb2[i]

        var w2_offset = layer * dim * hidden_dim
        matmul_simd_parallel(state.xb, 0, state.hb, 0, weights.w2, w2_offset, dim, hidden_dim)

        for i in range(dim):
            state.x[i] += state.xb[i]

    rmsnorm_simd(state.xb, 0, state.x, 0, weights.rms_final_weight, 0, dim)

    if config.shared_weights:
        matmul_simd_parallel(state.logits, 0, state.xb, 0, weights.token_embedding, 0, config.vocab_size, dim)
    else:
        matmul_simd_parallel(state.logits, 0, state.xb, 0, weights.wcls, 0, config.vocab_size, dim)


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
    var checkpoint = "stories15M.bin"
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

    print("Loading parallel SIMD model from", checkpoint)
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
