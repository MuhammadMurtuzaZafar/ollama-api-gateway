"""
EdgeLLM CUDA Inference - Mojo 25.5 Compatible
Generates actual text output using CUDA-accelerated kernels.

Compatible with:
- Mojo 25.5.0 (no global vars, uses alias instead of comptime)
- CUDA kernels via OwnedDLHandle FFI
- LLaMA/SmolLM model format

Usage:
    mojo build -O3 src/edgellm_inference.mojo -o bin/edgellm
    ./bin/edgellm models/stories15M.bin -z tokenizer.bin -i "Once upon a time"
"""
from collections import List, Dict
from sys import argv
from sys.ffi import OwnedDLHandle
from memory import UnsafePointer
import math
import random
import time

# Constants - use alias instead of comptime for Mojo 25.5
alias NUM_CONFIG_INT: Int = 7
alias EPS: Float32 = 1e-5


struct Matrix:
    """Matrix using List storage with direct indexing."""
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
    fn size(self) -> Int:
        return len(self.data)

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

    fn __init__(out self, path: String) raises:
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

        if self.vocab_size < 0:
            self.vocab_size = -self.vocab_size

        print("Config: dim=", self.dim, "hidden_dim=", self.hidden_dim)
        print("        n_layers=", self.n_layers, "n_heads=", self.n_heads)
        print("        vocab_size=", self.vocab_size, "seq_len=", self.seq_len)


struct Tokenizer:
    var vocab: List[String]
    var vocab_scores: List[Float32]
    var vocab_size: Int
    var max_token_length: Int
    var vocab_map: Dict[String, Int]

    fn __init__(out self, vocab_size: Int, path: String) raises:
        self.vocab_size = vocab_size
        self.vocab = List[String]()
        self.vocab_scores = List[Float32]()
        self.vocab_map = Dict[String, Int]()

        var f = open(path, "r")
        var max_len_bytes = f.read_bytes(4)
        self.max_token_length = Int(max_len_bytes.unsafe_ptr().bitcast[Int32]()[0])

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

    fn decode(self, token_id: Int) -> String:
        if token_id >= 0 and token_id < len(self.vocab):
            return self.vocab[token_id]
        return ""


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
    var shared_weights: Bool

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

        self.token_embedding = Matrix(
            read_weights(f, config.vocab_size * config.dim),
            config.vocab_size, config.dim
        )
        self.rms_att_weight = Matrix(
            read_weights(f, config.n_layers * config.dim),
            config.n_layers, config.dim
        )
        self.wq = Matrix(
            read_weights(f, config.n_layers * config.dim * config.dim),
            config.n_layers, config.dim, config.dim
        )
        self.wk = Matrix(
            read_weights(f, config.n_layers * config.kv_dim * config.dim),
            config.n_layers, config.kv_dim, config.dim
        )
        self.wv = Matrix(
            read_weights(f, config.n_layers * config.kv_dim * config.dim),
            config.n_layers, config.kv_dim, config.dim
        )
        self.wo = Matrix(
            read_weights(f, config.n_layers * config.dim * config.dim),
            config.n_layers, config.dim, config.dim
        )
        self.rms_ffn_weight = Matrix(
            read_weights(f, config.n_layers * config.dim),
            config.n_layers, config.dim
        )
        self.w1 = Matrix(
            read_weights(f, config.n_layers * config.hidden_dim * config.dim),
            config.n_layers, config.hidden_dim, config.dim
        )
        self.w2 = Matrix(
            read_weights(f, config.n_layers * config.dim * config.hidden_dim),
            config.n_layers, config.dim, config.hidden_dim
        )
        self.w3 = Matrix(
            read_weights(f, config.n_layers * config.hidden_dim * config.dim),
            config.n_layers, config.hidden_dim, config.dim
        )
        self.rms_final_weight = Matrix(
            read_weights(f, config.dim), config.dim
        )
        self.freq_cis_real = Matrix(
            read_weights(f, config.seq_len * config.head_size // 2),
            config.seq_len, config.head_size // 2
        )
        self.freq_cis_imag = Matrix(
            read_weights(f, config.seq_len * config.head_size // 2),
            config.seq_len, config.head_size // 2
        )

        # Initialize wcls - will be empty if weights are shared
        self.shared_weights = True
        self.wcls = Matrix(List[Float32](), 0, 0)

        f.close()
        print("Loaded", config.dim * config.n_layers // 1000, "K parameters")


struct RunState:
    var x: Matrix
    var xb: Matrix
    var xb2: Matrix
    var hb: Matrix
    var hb2: Matrix
    var q: Matrix
    var k: Matrix  # Temporary K for current position
    var v: Matrix  # Temporary V for current position
    var att: Matrix
    var logits: Matrix
    var key_cache: Matrix    # [n_layers * n_kv_heads, seq_len, head_size] for CUDA compatibility
    var value_cache: Matrix  # [n_layers * n_kv_heads, seq_len, head_size] for CUDA compatibility

    fn __init__(out self, config: Config):
        self.x = Matrix(config.dim)
        self.xb = Matrix(config.dim)
        self.xb2 = Matrix(config.dim)
        self.hb = Matrix(config.hidden_dim)
        self.hb2 = Matrix(config.hidden_dim)
        self.q = Matrix(config.dim)
        self.k = Matrix(config.kv_dim)  # Temp K
        self.v = Matrix(config.kv_dim)  # Temp V
        self.att = Matrix(config.n_heads, config.seq_len)
        self.logits = Matrix(config.vocab_size)
        # Cache layout: [n_layers * n_kv_heads, seq_len, head_size]
        # This is compatible with CUDA kernel expecting [n_heads, cache_len, head_dim]
        self.key_cache = Matrix(config.n_layers * config.n_kv_heads, config.seq_len, config.head_size)
        self.value_cache = Matrix(config.n_layers * config.n_kv_heads, config.seq_len, config.head_size)


struct CUDAKernels:
    """CUDA kernel handles for GPU acceleration."""
    var flash_attn_path: String
    var rmsnorm_path: String
    var ffn_path: String
    var available: Bool

    fn __init__(out self, lib_dir: String):
        self.available = False
        self.flash_attn_path = lib_dir + "/libflash_attention_int8.so"
        self.rmsnorm_path = lib_dir + "/librmsnorm_kernel.so"
        self.ffn_path = lib_dir + "/libffn_kernel.so"
        # Actual loading happens in try_load()

    fn try_load(mut self) raises -> Bool:
        """Try to load CUDA kernel libraries."""
        try:
            var test = OwnedDLHandle(self.flash_attn_path)
            _ = test
            self.available = True
            print("CUDA kernels available at:", self.flash_attn_path)
            return True
        except:
            print("CUDA kernels not available - using CPU fallback")
            return False


# =============================================================================
# Core operations
# =============================================================================

fn rmsnorm(mut out_mat: Matrix, out_offset: Int,
           x_mat: Matrix, x_offset: Int,
           w_mat: Matrix, w_offset: Int, size: Int):
    """RMS normalization."""
    var ss: Float32 = 0.0
    for i in range(size):
        var v = x_mat[x_offset + i]
        ss += v * v
    ss = 1.0 / math.sqrt(ss / size + EPS)
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


fn matmul(mut out_mat: Matrix, out_offset: Int,
          x_mat: Matrix, x_offset: Int,
          w_mat: Matrix, w_offset: Int, rows: Int, cols: Int):
    """Matrix-vector multiplication."""
    for i in range(rows):
        var sum: Float32 = 0.0
        var row_off = w_offset + i * cols
        for j in range(cols):
            sum += x_mat[x_offset + j] * w_mat[row_off + j]
        out_mat[out_offset + i] = sum


fn transformer_forward(
    token: Int,
    pos: Int,
    config: Config,
    mut state: RunState,
    weights: TransformerWeights,
) raises:
    """Forward pass of the transformer with CUDA-compatible cache layout."""
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var head_size = config.head_size
    var kv_dim = config.kv_dim
    var kv_mul = config.kv_mul
    var n_heads = config.n_heads
    var n_kv_heads = config.n_kv_heads
    var seq_len = config.seq_len
    var sqrt_head_size = math.sqrt(Float32(head_size))

    # Copy token embedding into x
    var emb_offset = token * dim
    for i in range(dim):
        state.x[i] = weights.token_embedding[emb_offset + i]

    # Frequency components for RoPE
    var freq_offset = pos * (head_size // 2)

    # Forward through layers
    for layer in range(config.n_layers):
        var layer_dim_offset = layer * dim

        # Attention rmsnorm
        rmsnorm(state.xb, 0, state.x, 0,
                weights.rms_att_weight, layer_dim_offset, dim)

        # QKV projections - Q, K, V to temp buffers
        var wq_offset = layer * dim * dim
        var wk_offset = layer * kv_dim * dim
        var wv_offset = layer * kv_dim * dim

        matmul(state.q, 0, state.xb, 0, weights.wq, wq_offset, dim, dim)
        matmul(state.k, 0, state.xb, 0, weights.wk, wk_offset, kv_dim, dim)
        matmul(state.v, 0, state.xb, 0, weights.wv, wv_offset, kv_dim, dim)

        # Apply RoPE to Q and K (in temp buffers)
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
                    var k_idx = h * head_size + j
                    var k0 = state.k[k_idx]
                    var k1 = state.k[k_idx + 1]
                    state.k[k_idx] = k0 * fcr - k1 * fci
                    state.k[k_idx + 1] = k0 * fci + k1 * fcr

        # Copy K, V to cache with CUDA-compatible layout [n_kv_heads, seq_len, head_size]
        for h in range(n_kv_heads):
            var cache_head_offset = (layer * n_kv_heads + h) * seq_len * head_size + pos * head_size
            var src_offset = h * head_size
            for i in range(head_size):
                state.key_cache[cache_head_offset + i] = state.k[src_offset + i]
                state.value_cache[cache_head_offset + i] = state.v[src_offset + i]

        # Zero xb for attention output
        state.xb.zero()

        # Multihead attention (CPU path)
        for h in range(n_heads):
            var q_offset = h * head_size
            var att_offset = h * seq_len

            # Which KV head does this Q head use? (for GQA)
            var kv_head = h // kv_mul
            var kv_head_cache_offset = (layer * n_kv_heads + kv_head) * seq_len * head_size

            # Compute attention scores for all positions
            for t in range(pos + 1):
                var k_offset = kv_head_cache_offset + t * head_size

                var score: Float32 = 0.0
                for i in range(head_size):
                    score += state.q[q_offset + i] * state.key_cache[k_offset + i]
                state.att[att_offset + t] = score / sqrt_head_size

            # Softmax attention scores
            softmax(state.att, att_offset, pos + 1)

            # Weighted sum of values
            var xb_offset = h * head_size
            for t in range(pos + 1):
                var v_offset = kv_head_cache_offset + t * head_size
                var a = state.att[att_offset + t]
                for i in range(head_size):
                    state.xb[xb_offset + i] += a * state.value_cache[v_offset + i]

        # Output projection
        var wo_offset = layer * dim * dim
        matmul(state.xb2, 0, state.xb, 0, weights.wo, wo_offset, dim, dim)

        # Residual connection
        for i in range(dim):
            state.x[i] += state.xb2[i]

        # FFN rmsnorm
        rmsnorm(state.xb, 0, state.x, 0,
                weights.rms_ffn_weight, layer_dim_offset, dim)

        # FFN: w1 and w3
        var w1_offset = layer * hidden_dim * dim
        var w3_offset = layer * hidden_dim * dim
        matmul(state.hb, 0, state.xb, 0, weights.w1, w1_offset, hidden_dim, dim)
        matmul(state.hb2, 0, state.xb, 0, weights.w3, w3_offset, hidden_dim, dim)

        # SiLU and element-wise multiply
        for i in range(hidden_dim):
            var val = state.hb[i]
            state.hb[i] = val * (1.0 / (1.0 + math.exp(-val))) * state.hb2[i]

        # w2 projection
        var w2_offset = layer * dim * hidden_dim
        matmul(state.xb, 0, state.hb, 0, weights.w2, w2_offset, dim, hidden_dim)

        # Residual connection
        for i in range(dim):
            state.x[i] += state.xb[i]

    # Final rmsnorm (use xb as temp to avoid aliasing)
    rmsnorm(state.xb, 0, state.x, 0, weights.rms_final_weight, 0, dim)

    # Classifier - use token embedding if shared weights
    if weights.shared_weights:
        matmul(state.logits, 0, state.xb, 0, weights.token_embedding, 0, config.vocab_size, dim)
    else:
        matmul(state.logits, 0, state.xb, 0, weights.wcls, 0, config.vocab_size, dim)


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
    """Encode text using BPE."""
    for i in range(len(text)):
        var c = String(text[i])
        var idx = tok.find(c)
        if idx == -1:
            print("Unknown character at position", i)
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
    elif len(s) > 0 and s[0] == '<' and s[len(s)-1] == '>':
        # Skip special tokens like <s>, </s>, <unk>
        pass
    else:
        print(s, end="")


struct CUDAInference:
    """CUDA-accelerated inference using stateless attention kernel."""
    var flash_attn: OwnedDLHandle
    var available: Bool
    var initialized: Bool
    var n_heads: Int
    var max_seq: Int
    var head_dim: Int

    fn __init__(out self, lib_dir: String) raises:
        self.available = False
        self.initialized = False
        self.n_heads = 0
        self.max_seq = 0
        self.head_dim = 0

        # Try to load CUDA libraries
        try:
            self.flash_attn = OwnedDLHandle(lib_dir + "/libflash_attention_int8.so")
            self.available = True
            print("CUDA library loaded:", lib_dir + "/libflash_attention_int8.so")
        except:
            self.flash_attn = OwnedDLHandle("")
            print("CUDA library not available")

    fn init_stateless_attention(mut self, n_heads: Int, max_seq: Int, head_dim: Int) -> Bool:
        """Initialize stateless attention (no internal KV cache)."""
        if not self.available:
            return False

        var ret = self.flash_attn.call["attention_stateless_init", Int32](
            Int32(n_heads), Int32(max_seq), Int32(head_dim))

        if ret == 0:
            self.initialized = True
            self.n_heads = n_heads
            self.max_seq = max_seq
            self.head_dim = head_dim
            print("Stateless Attention initialized: heads=", n_heads, "seq=", max_seq, "dim=", head_dim)
            return True
        else:
            print("Stateless Attention init failed")
            return False

    fn attention_strided(self,
                 q_ptr: UnsafePointer[Float32],
                 k_cache_ptr: UnsafePointer[Float32],
                 v_cache_ptr: UnsafePointer[Float32],
                 o_ptr: UnsafePointer[Float32],
                 n_heads: Int,
                 cache_len: Int,
                 head_dim: Int,
                 buffer_seq_len: Int) -> Int:
        """Execute strided FP32 attention with external KV cache.

        Use when KV cache has [n_heads, max_seq_len, head_dim] layout
        but only cache_len positions are valid.
        """
        if not self.initialized:
            return -1

        # Use pure FP32 strided version (no INT8 quantization issues)
        return Int(self.flash_attn.call["attention_fp32_strided", Int32](
            q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,
            Int32(n_heads), Int32(cache_len), Int32(head_dim), Int32(buffer_seq_len)))

    fn cleanup(mut self):
        """Cleanup CUDA resources."""
        if self.initialized:
            self.flash_attn.call["attention_stateless_cleanup", NoneType]()
            self.initialized = False


fn transformer_forward_cuda(
    token: Int,
    pos: Int,
    config: Config,
    mut state: RunState,
    weights: TransformerWeights,
    cuda: CUDAInference,
) raises:
    """CUDA-accelerated forward pass using stateless attention with per-layer KV caches."""
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var head_size = config.head_size
    var kv_dim = config.kv_dim
    var kv_mul = config.kv_mul
    var n_heads = config.n_heads
    var n_kv_heads = config.n_kv_heads
    var seq_len = config.seq_len
    var sqrt_head_size = math.sqrt(Float32(head_size))

    # Copy token embedding into x
    var emb_offset = token * dim
    for i in range(dim):
        state.x[i] = weights.token_embedding[emb_offset + i]

    # Frequency components for RoPE
    var freq_offset = pos * (head_size // 2)

    # Forward through layers
    for layer in range(config.n_layers):
        var layer_dim_offset = layer * dim

        # Attention rmsnorm
        rmsnorm(state.xb, 0, state.x, 0,
                weights.rms_att_weight, layer_dim_offset, dim)

        # QKV projections - Q, K, V to temp buffers
        var wq_offset = layer * dim * dim
        var wk_offset = layer * kv_dim * dim
        var wv_offset = layer * kv_dim * dim

        matmul(state.q, 0, state.xb, 0, weights.wq, wq_offset, dim, dim)
        matmul(state.k, 0, state.xb, 0, weights.wk, wk_offset, kv_dim, dim)
        matmul(state.v, 0, state.xb, 0, weights.wv, wv_offset, kv_dim, dim)

        # Apply RoPE to Q and K (in temp buffers)
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
                    var k_idx = h * head_size + j
                    var k0 = state.k[k_idx]
                    var k1 = state.k[k_idx + 1]
                    state.k[k_idx] = k0 * fcr - k1 * fci
                    state.k[k_idx + 1] = k0 * fci + k1 * fcr

        # Copy K, V to cache with CUDA-compatible layout [n_kv_heads, seq_len, head_size]
        for h in range(n_kv_heads):
            var cache_head_offset = (layer * n_kv_heads + h) * seq_len * head_size + pos * head_size
            var src_offset = h * head_size
            for i in range(head_size):
                state.key_cache[cache_head_offset + i] = state.k[src_offset + i]
                state.value_cache[cache_head_offset + i] = state.v[src_offset + i]

        # Use CUDA stateless attention with per-layer KV caches
        if cuda.initialized:
            # Get pointers for this layer's KV cache
            var layer_cache_offset = layer * n_kv_heads * seq_len * head_size
            var q_ptr = state.q.data.unsafe_ptr()
            var k_cache_ptr = state.key_cache.data.unsafe_ptr() + layer_cache_offset
            var v_cache_ptr = state.value_cache.data.unsafe_ptr() + layer_cache_offset
            var o_ptr = state.xb.data.unsafe_ptr()

            # Call strided stateless attention - handles [n_heads, seq_len, head_dim] layout
            # where only (pos+1) positions are valid but stride is seq_len
            # Note: For GQA (n_heads != n_kv_heads), we pass n_kv_heads to kernel
            # and handle head mapping on output side
            var ret = cuda.attention_strided(q_ptr, k_cache_ptr, v_cache_ptr, o_ptr,
                                     n_kv_heads, pos + 1, head_size, seq_len)

            if ret != 0:
                # Fallback to CPU attention if CUDA fails
                state.xb.zero()
                for h in range(n_heads):
                    var q_offset = h * head_size
                    var att_offset = h * seq_len
                    var kv_head = h // kv_mul
                    var kv_head_cache_offset = (layer * n_kv_heads + kv_head) * seq_len * head_size

                    for t in range(pos + 1):
                        var k_offset = kv_head_cache_offset + t * head_size
                        var score: Float32 = 0.0
                        for i in range(head_size):
                            score += state.q[q_offset + i] * state.key_cache[k_offset + i]
                        state.att[att_offset + t] = score / sqrt_head_size

                    softmax(state.att, att_offset, pos + 1)

                    var xb_offset = h * head_size
                    for t in range(pos + 1):
                        var v_offset = kv_head_cache_offset + t * head_size
                        var a = state.att[att_offset + t]
                        for i in range(head_size):
                            state.xb[xb_offset + i] += a * state.value_cache[v_offset + i]
            else:
                # CUDA kernel outputs [n_kv_heads, head_size], need to expand for GQA
                if n_heads != n_kv_heads:
                    # GQA: Copy each KV head output to multiple Q heads
                    for h in range(n_heads - 1, -1, -1):  # Reverse to avoid overwrite
                        var kv_head = h // kv_mul
                        var src_offset = kv_head * head_size
                        var dst_offset = h * head_size
                        if src_offset != dst_offset:
                            for i in range(head_size):
                                state.xb[dst_offset + i] = state.xb[src_offset + i]
        else:
            # CPU attention fallback
            state.xb.zero()
            for h in range(n_heads):
                var q_offset = h * head_size
                var att_offset = h * seq_len
                var kv_head = h // kv_mul
                var kv_head_cache_offset = (layer * n_kv_heads + kv_head) * seq_len * head_size

                for t in range(pos + 1):
                    var k_offset = kv_head_cache_offset + t * head_size
                    var score: Float32 = 0.0
                    for i in range(head_size):
                        score += state.q[q_offset + i] * state.key_cache[k_offset + i]
                    state.att[att_offset + t] = score / sqrt_head_size

                softmax(state.att, att_offset, pos + 1)

                var xb_offset = h * head_size
                for t in range(pos + 1):
                    var v_offset = kv_head_cache_offset + t * head_size
                    var a = state.att[att_offset + t]
                    for i in range(head_size):
                        state.xb[xb_offset + i] += a * state.value_cache[v_offset + i]

        # Output projection
        var wo_offset = layer * dim * dim
        matmul(state.xb2, 0, state.xb, 0, weights.wo, wo_offset, dim, dim)

        # Residual connection
        for i in range(dim):
            state.x[i] += state.xb2[i]

        # FFN rmsnorm
        rmsnorm(state.xb, 0, state.x, 0,
                weights.rms_ffn_weight, layer_dim_offset, dim)

        # FFN: w1 and w3
        var w1_offset = layer * hidden_dim * dim
        var w3_offset = layer * hidden_dim * dim
        matmul(state.hb, 0, state.xb, 0, weights.w1, w1_offset, hidden_dim, dim)
        matmul(state.hb2, 0, state.xb, 0, weights.w3, w3_offset, hidden_dim, dim)

        # SiLU and element-wise multiply
        for i in range(hidden_dim):
            var val = state.hb[i]
            state.hb[i] = val * (1.0 / (1.0 + math.exp(-val))) * state.hb2[i]

        # w2 projection
        var w2_offset = layer * dim * hidden_dim
        matmul(state.xb, 0, state.hb, 0, weights.w2, w2_offset, dim, hidden_dim)

        # Residual connection
        for i in range(dim):
            state.x[i] += state.xb[i]

    # Final rmsnorm
    rmsnorm(state.xb, 0, state.x, 0, weights.rms_final_weight, 0, dim)

    # Classifier
    if weights.shared_weights:
        matmul(state.logits, 0, state.xb, 0, weights.token_embedding, 0, config.vocab_size, dim)
    else:
        matmul(state.logits, 0, state.xb, 0, weights.wcls, 0, config.vocab_size, dim)


fn main() raises:
    var checkpoint = "stories15M.bin"
    var tokenizer_path = "tokenizer.bin"
    var temperature: Float32 = 0.9
    var steps = 256
    var prompt = String("")
    var use_cuda = False
    var lib_dir = "./lib"
    var bos_token = 1  # Default: SentencePiece BOS
    var eos_token = 2  # Default: SentencePiece EOS

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
        elif args[i] == "--cuda":
            use_cuda = True
            i += 1
        elif args[i] == "--lib" and i + 1 < len(args):
            lib_dir = args[i + 1]
            i += 2
        elif args[i] == "--eos" and i + 1 < len(args):
            eos_token = atol(args[i + 1])
            i += 2
        elif args[i] == "--bos" and i + 1 < len(args):
            bos_token = atol(args[i + 1])
            i += 2
        elif args[i] == "--qwen":
            # Qwen uses EOS=151643, no BOS (use 0)
            eos_token = 151643
            bos_token = 0
            i += 1
        else:
            i += 1

    print()
    print("=" * 60)
    print("EdgeLLM Inference - Mojo 25.5")
    print("=" * 60)
    print()
    print("Loading model from", checkpoint)

    var config = Config(checkpoint)
    var weights = TransformerWeights(checkpoint, config)
    var tokenizer = Tokenizer(config.vocab_size, tokenizer_path)
    var state = RunState(config)

    # Initialize CUDA if requested (using stateless attention for per-layer KV caches)
    var cuda = CUDAInference(lib_dir)
    if use_cuda and cuda.available:
        # Use n_kv_heads since that's what the kernel operates on
        var ok = cuda.init_stateless_attention(config.n_kv_heads, config.seq_len, config.head_size)
        if ok:
            print("Backend: CUDA (Stateless Attention)")
        else:
            use_cuda = False
            print("Backend: CPU (CUDA init failed)")
    else:
        if use_cuda:
            print("Backend: CPU (CUDA not available)")
        else:
            print("Backend: CPU")

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    var prompt_tokens = List[Int]()
    if len(prompt) > 0:
        bpe_encode(prompt_tokens, prompt, tokenizer)
        print("Prompt:", prompt)
        print("Prompt tokens:", len(prompt_tokens))

    print()
    print("Generating", steps, "tokens...")
    print("Temperature:", temperature)
    print("BOS token:", bos_token, "EOS token:", eos_token)
    print("-" * 60)
    print()

    # Start with BOS token (or first prompt token if BOS=0/no BOS)
    var token = bos_token
    if bos_token == 0 and len(prompt_tokens) > 0:
        token = prompt_tokens[0]  # Use first prompt token if no BOS

    var start_time = time.perf_counter_ns()
    var tokens_generated = 0

    for pos in range(steps):
        if use_cuda and cuda.initialized:
            transformer_forward_cuda(token, pos, config, state, weights, cuda)
        else:
            transformer_forward(token, pos, config, state, weights)

        var next_token: Int
        # For models without BOS (like Qwen), prompt starts at pos 0
        # For models with BOS, prompt starts at pos 1
        var prompt_offset = 1 if bos_token != 0 else 0
        if pos + prompt_offset < len(prompt_tokens):
            next_token = prompt_tokens[pos + prompt_offset]
        else:
            if temperature == 0.0:
                next_token = argmax(state.logits, config.vocab_size)
            else:
                for j in range(config.vocab_size):
                    state.logits[j] /= temperature
                softmax(state.logits, 0, config.vocab_size)
                next_token = sample(state.logits, config.vocab_size)

        # Check for EOS token (configurable via --eos or --qwen flags)
        if next_token == eos_token:
            break

        print_token(tokenizer, next_token)
        token = next_token
        tokens_generated += 1

    var end_time = time.perf_counter_ns()
    var elapsed_ms = (end_time - start_time) // 1_000_000

    # Cleanup CUDA
    if use_cuda and cuda.initialized:
        cuda.cleanup()

    print()
    print()
    print("-" * 60)
    print("Generated", tokens_generated, "tokens in", elapsed_ms, "ms")
    if elapsed_ms > 0:
        var tok_per_sec = tokens_generated * 1000 // Int(elapsed_ms)
        print("Speed:", tok_per_sec, "tokens/sec")
    if use_cuda:
        print("Backend: CUDA (Stateless Attention)")
