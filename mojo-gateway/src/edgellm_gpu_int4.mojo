"""
EdgeLLM INT4 GPU Inference - 400+ tok/s Target

Key optimizations:
1. INT4 quantized weights (8x memory reduction)
2. On-the-fly dequantization in custom GEMV kernel
3. FP32 embeddings and norms for quality
4. Group-wise scaling (GROUP_SIZE=128, AWQ-style)
"""

from collections import List, Dict
from sys import argv
from sys.ffi import OwnedDLHandle
from memory import UnsafePointer
import time

alias NUM_CONFIG_INT: Int = 8  # INT4 header has 8 fields (including is_int4 flag)
alias GROUP_SIZE: Int = 128
alias EPS: Float32 = 1e-6


struct ConfigINT4:
    """Configuration for INT4 quantized model."""
    var dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var vocab_size: Int
    var seq_len: Int
    var is_int4: Int  # 1 = INT4, 0 = FP32
    var head_size: Int
    var kv_dim: Int
    var kv_mul: Int
    var n_groups_dim: Int
    var n_groups_hd: Int

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
        self.is_int4 = Int(ptr[7])

        self.head_size = self.dim // self.n_heads
        self.kv_dim = (self.n_kv_heads * self.dim) // self.n_heads
        self.kv_mul = self.n_heads // self.n_kv_heads
        self.n_groups_dim = (self.dim + GROUP_SIZE - 1) // GROUP_SIZE
        self.n_groups_hd = (self.hidden_dim + GROUP_SIZE - 1) // GROUP_SIZE

        if self.vocab_size < 0:
            self.vocab_size = -self.vocab_size


struct Tokenizer:
    var vocab: List[String]
    var vocab_scores: List[Float32]
    var vocab_map: Dict[String, Int]
    var vocab_size: Int
    var max_token_length: Int

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


struct GPUInferenceINT4:
    """INT4 quantized GPU inference using custom GEMV kernels."""
    var handle: OwnedDLHandle
    var available: Bool
    var configured: Bool

    fn __init__(out self, lib_path: String) raises:
        self.available = False
        self.configured = False
        self.handle = OwnedDLHandle(lib_path)
        self.available = True

    fn init_int4(mut self, model_path: String, config: ConfigINT4) raises -> Bool:
        """Initialize INT4 GPU and upload weights."""
        if not self.available:
            return False

        # Calculate buffer sizes
        var n = config.n_layers
        var d = config.dim
        var hd = config.hidden_dim
        var kv = config.kv_dim
        var v = config.vocab_size
        var s = config.seq_len
        var hs = config.head_size
        var ng_dim = config.n_groups_dim
        var ng_hd = config.n_groups_hd

        # FP32 sizes (embeddings, norms, biases, RoPE)
        var fp32_elements = (
            v * d +           # token_embedding
            n * d +           # rms_att
            n * d +           # rms_ffn
            d +               # rms_final
            s * hs // 2 +     # freq_cos
            s * hs // 2 +     # freq_sin
            n * d +           # bq
            n * kv +          # bk
            n * kv            # bv
        )
        var fp32_bytes = fp32_elements * 4

        # INT4 packed weights sizes (in bytes)
        var wq_packed = n * d * d // 2
        var wk_packed = n * kv * d // 2
        var wv_packed = n * kv * d // 2
        var wo_packed = n * d * d // 2
        var w1_packed = n * hd * d // 2
        var w2_packed = n * d * hd // 2
        var w3_packed = n * hd * d // 2
        var int4_packed_bytes = wq_packed + wk_packed + wv_packed + wo_packed + w1_packed + w2_packed + w3_packed

        # INT4 scales sizes (FP16 = 2 bytes each)
        var wq_scales = n * d * ng_dim * 2
        var wk_scales = n * kv * ng_dim * 2
        var wv_scales = n * kv * ng_dim * 2
        var wo_scales = n * d * ng_dim * 2
        var w1_scales = n * hd * ng_dim * 2
        var w2_scales = n * d * ng_hd * 2
        var w3_scales = n * hd * ng_dim * 2
        var int4_scales_bytes = wq_scales + wk_scales + wv_scales + wo_scales + w1_scales + w2_scales + w3_scales

        # Activation buffer size
        var act_elements = (
            d +               # x
            d +               # xb
            d +               # xb2
            d +               # q
            kv +              # k
            kv +              # v
            hd +              # hb
            hd +              # hb2
            v +               # logits
            n * config.n_kv_heads * s * hs +  # k_cache
            n * config.n_kv_heads * s * hs +  # v_cache
            1                 # result
        )
        var act_bytes = act_elements * 4

        print("INT4 model sizes:")
        print("  FP32 (embedding/norms):", fp32_bytes // 1024 // 1024, "MB")
        print("  INT4 packed weights:", int4_packed_bytes // 1024 // 1024, "MB")
        print("  INT4 scales:", int4_scales_bytes // 1024 // 1024, "MB")
        print("  Total model:", (fp32_bytes + int4_packed_bytes + int4_scales_bytes) // 1024 // 1024, "MB")
        print("  Activations:", act_bytes // 1024 // 1024, "MB")

        # Initialize INT4 cuBLAS
        var ret = self.handle.call["cublas_init_int4", Int32](
            Int64(fp32_bytes), Int64(int4_packed_bytes), Int64(int4_scales_bytes), Int64(act_bytes)
        )
        if ret != 0:
            print("Failed to init INT4 cuBLAS")
            return False

        # Read and upload weights from file
        if not self.load_int4_weights(model_path, config, fp32_bytes, int4_packed_bytes, int4_scales_bytes):
            print("Failed to load INT4 weights")
            return False

        # Configure GPU
        ret = self.handle.call["gpu_configure_int4", Int32](
            Int32(d), Int32(hd), Int32(n), Int32(config.n_heads),
            Int32(config.n_kv_heads), Int32(v), Int32(s), Int32(1)  # has_bias = 1
        )
        if ret != 0:
            print("Failed to configure INT4 GPU")
            return False

        self.configured = True
        print("INT4 GPU initialized and configured")
        return True

    fn load_int4_weights(self, path: String, config: ConfigINT4,
                        fp32_bytes: Int, int4_packed_bytes: Int, int4_scales_bytes: Int) raises -> Bool:
        """Load INT4 model weights from file and upload to GPU."""
        var f = open(path, "r")

        # Skip header (8 int32 = 32 bytes)
        _ = f.read_bytes(32)

        # Read FP32 data (embedding, norms, biases)
        var fp32_data = f.read_bytes(fp32_bytes)
        var fp32_ptr = fp32_data.unsafe_ptr().bitcast[Float32]()

        # Read INT4 scales (FP16)
        var scales_data = f.read_bytes(int4_scales_bytes)
        var scales_ptr = scales_data.unsafe_ptr()

        # Read INT4 packed weights
        var packed_data = f.read_bytes(int4_packed_bytes)
        var packed_ptr = packed_data.unsafe_ptr()

        f.close()

        # Upload to GPU
        var ret = self.handle.call["cublas_upload_int4_weights", Int32](
            fp32_ptr, Int64(fp32_bytes),
            packed_ptr, Int64(int4_packed_bytes),
            scales_ptr, Int64(int4_scales_bytes)
        )

        return ret == 0

    fn forward(self, token: Int, pos: Int) -> Int:
        """INT4 forward pass. Returns next token."""
        if not self.configured:
            return -1
        return Int(self.handle.call["gpu_forward_int4", Int32](Int32(token), Int32(pos)))

    fn load_int8_embedding(self, path: String, vocab_size: Int, dim: Int) raises -> Bool:
        """Load INT8 embedding for fast logit computation (optional)."""
        if not self.available:
            return False

        # Initialize INT8 embedding buffers
        var ret = self.handle.call["int8_embedding_init", Int32](Int32(vocab_size), Int32(dim))
        if ret != 0:
            print("Failed to init INT8 embedding")
            return False

        # Read INT8 embedding file
        var f = open(path, "r")

        # Read header: vocab_size, dim
        var header_bytes = f.read_bytes(8)
        var header_ptr = header_bytes.unsafe_ptr().bitcast[Int32]()
        var file_vocab = Int(header_ptr[0])
        var file_dim = Int(header_ptr[1])

        if file_vocab != vocab_size or file_dim != dim:
            print("INT8 embedding file mismatch: expected", vocab_size, "x", dim, "got", file_vocab, "x", file_dim)
            f.close()
            return False

        # Read INT8 embedding data
        var emb_bytes = vocab_size * dim
        var emb_data = f.read_bytes(emb_bytes)
        var emb_ptr = emb_data.unsafe_ptr().bitcast[Int8]()

        # Read FP16 scales
        var scales_bytes = vocab_size * 2  # FP16 = 2 bytes
        var scales_data = f.read_bytes(scales_bytes)
        var scales_ptr = scales_data.unsafe_ptr()

        f.close()

        # Upload to GPU
        ret = self.handle.call["int8_embedding_upload", Int32](
            emb_ptr, scales_ptr, Int32(vocab_size), Int32(dim)
        )
        if ret != 0:
            print("Failed to upload INT8 embedding")
            return False

        # Enable INT8 embedding mode
        self.handle.call["set_int8_embedding_mode", NoneType](Int32(1))

        print("INT8 embedding loaded: ~3x faster logit computation enabled")
        return True

    fn cleanup(self):
        if self.available:
            self.handle.call["cublas_cleanup_int4", NoneType]()


fn bpe_encode(mut tokens: List[Int], text: String, tok: Tokenizer):
    """Encode text using BPE with merging."""
    # Step 1: Tokenize character by character
    for i in range(len(text)):
        var c = String(text[i])
        var idx = tok.find(c)
        if idx == -1:
            # Try linear search as fallback
            for j in range(len(tok.vocab)):
                if tok.vocab[j] == c:
                    idx = j
                    break
        if idx != -1:
            tokens.append(idx)

    # Step 2: Iteratively merge adjacent tokens based on scores
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

        # Replace the two tokens with the merged token
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
        pass
    else:
        print(s, end="")


fn main() raises:
    var checkpoint = "models/qwen2.5-1.5b_int4.bin"
    var tokenizer_path = "models/qwen2.5-1.5b_int4_tokenizer.bin"
    var steps = 100
    var prompt = String("Hello")
    var lib_path = "./lib/libcublas_int4.so"
    var eos_token = 151643
    var bos_token = 0

    var args = argv()
    var i = 1
    while i < len(args):
        if args[i] == "-m" and i + 1 < len(args):
            checkpoint = args[i + 1]
            i += 2
        elif args[i] == "-z" and i + 1 < len(args):
            tokenizer_path = args[i + 1]
            i += 2
        elif args[i] == "-n" and i + 1 < len(args):
            steps = atol(args[i + 1])
            i += 2
        elif args[i] == "-i" and i + 1 < len(args):
            prompt = args[i + 1]
            i += 2
        elif args[i] == "--lib" and i + 1 < len(args):
            lib_path = args[i + 1]
            i += 2
        else:
            i += 1

    print()
    print("=" * 60)
    print("EdgeLLM INT4 GPU Inference - 400+ tok/s Target")
    print("=" * 60)
    print()

    # Load config
    print("Loading INT4 model config from", checkpoint)
    var config = ConfigINT4(checkpoint)

    if config.is_int4 != 1:
        print("ERROR: Model is not INT4 quantized (is_int4 =", config.is_int4, ")")
        print("Use export_qwen_int4.py to create an INT4 model")
        return

    print("Config:")
    print("  dim =", config.dim)
    print("  hidden_dim =", config.hidden_dim)
    print("  n_layers =", config.n_layers)
    print("  n_heads =", config.n_heads, "n_kv_heads =", config.n_kv_heads)
    print("  vocab_size =", config.vocab_size)
    print("  seq_len =", config.seq_len)
    print("  is_int4 =", config.is_int4, "(INT4 quantized)")
    print("  group_size =", GROUP_SIZE)
    print()

    # Initialize INT4 GPU
    print("Initializing INT4 GPU...")
    var gpu = GPUInferenceINT4(lib_path)

    if not gpu.available:
        print("ERROR: INT4 GPU library not available at", lib_path)
        print("Build with: make -C src/kernels/cuda cublas-int4")
        return

    # Initialize and upload weights
    print("Uploading INT4 weights to GPU...")
    if not gpu.init_int4(checkpoint, config):
        print("ERROR: Failed to initialize INT4 GPU")
        return

    # Try to load INT8 embedding for fast logit computation
    var int8_emb_path = checkpoint.replace(".bin", "_int8_emb.bin")
    print("Checking for INT8 embedding:", int8_emb_path)
    try:
        var f = open(int8_emb_path, "r")
        f.close()
        print("Found INT8 embedding file, loading...")
        if not gpu.load_int8_embedding(int8_emb_path, config.vocab_size, config.dim):
            print("Warning: INT8 embedding load failed, using FP32 cuBLAS (slower)")
    except:
        print("No INT8 embedding file found, using FP32 cuBLAS for logits")

    # Load tokenizer
    print("Loading tokenizer...")
    var tokenizer = Tokenizer(config.vocab_size, tokenizer_path)

    # Encode prompt
    var prompt_tokens = List[Int]()
    if len(prompt) > 0:
        bpe_encode(prompt_tokens, prompt, tokenizer)
    print("Prompt:", prompt)
    print("Prompt tokens:", len(prompt_tokens))

    print()
    print("Generating", steps, "tokens...")
    print("-" * 60)
    print()

    # Generate
    var token = bos_token
    if bos_token == 0 and len(prompt_tokens) > 0:
        token = prompt_tokens[0]

    var start_time = time.perf_counter_ns()
    var tokens_generated = 0

    for pos in range(steps):
        var next_token = gpu.forward(token, pos)

        if next_token < 0:
            print("\nERROR: Forward pass failed")
            break

        # Use prompt token if still in prompt
        var prompt_idx = pos + 1 if bos_token == 0 else pos
        if prompt_idx < len(prompt_tokens):
            next_token = prompt_tokens[prompt_idx]

        if next_token == eos_token:
            break

        print_token(tokenizer, next_token)
        token = next_token
        tokens_generated += 1

    var end_time = time.perf_counter_ns()
    var elapsed_ms = (end_time - start_time) // 1_000_000

    gpu.cleanup()

    print()
    print()
    print("-" * 60)
    print("Generated", tokens_generated, "tokens in", elapsed_ms, "ms")
    if elapsed_ms > 0:
        var tok_per_sec = tokens_generated * 1000 // Int(elapsed_ms)
        print("Speed:", tok_per_sec, "tokens/sec")
    print("=" * 60)
