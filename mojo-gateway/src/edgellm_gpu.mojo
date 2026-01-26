"""
EdgeLLM Full GPU Inference - 400+ tok/s Target

Key optimizations:
1. ALL weights on GPU (uploaded once)
2. ALL compute on GPU via cuBLAS
3. GQA-aware attention kernel
4. Minimal CPU-GPU transfers (only token IDs in, token IDs out)
"""

from collections import List, Dict
from sys import argv
from sys.ffi import OwnedDLHandle
from memory import UnsafePointer
import time

alias NUM_CONFIG_INT: Int = 7
alias EPS: Float32 = 1e-6


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


struct GPUInference:
    """Full GPU inference using cuBLAS."""
    var handle: OwnedDLHandle
    var available: Bool
    var configured: Bool

    fn __init__(out self, lib_path: String) raises:
        self.available = False
        self.configured = False
        self.handle = OwnedDLHandle(lib_path)
        self.available = True

    fn init_and_upload(mut self, model_path: String, config: Config) raises -> Bool:
        """Initialize GPU and upload model weights."""
        if not self.available:
            return False

        # Calculate sizes
        var n = config.n_layers
        var d = config.dim
        var hd = config.hidden_dim
        var kv = config.kv_dim
        var v = config.vocab_size
        var s = config.seq_len
        var hs = config.head_size

        # Base weights size
        var base_weights = (
            v * d +           # token_embedding
            n * d +           # rms_att
            n * d * d +       # wq
            n * kv * d +      # wk
            n * kv * d +      # wv
            n * d * d +       # wo
            n * d +           # rms_ffn
            n * hd * d +      # w1
            n * d * hd +      # w2
            n * hd * d +      # w3
            d +               # rms_final
            s * hs // 2 +     # freq_cos
            s * hs // 2       # freq_sin
        )

        # Bias weights (for Qwen)
        var bias_weights = n * d + n * kv + n * kv

        # Activation buffer size
        var act_size = (
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

        var weight_bytes = (base_weights + bias_weights) * 4
        var act_bytes = act_size * 4

        # Initialize cuBLAS and allocate memory
        var ret = self.handle.call["cublas_init", Int32](Int64(weight_bytes), Int64(act_bytes))
        if ret != 0:
            print("Failed to init cuBLAS")
            return False

        # Read weights from file
        var f = open(model_path, "r")
        _ = f.read_bytes(NUM_CONFIG_INT * 4)  # Skip config header

        # Read all weights (base + potential biases)
        var total_weight_size = base_weights + bias_weights
        var weight_data = f.read_bytes(total_weight_size * 4)
        var weight_ptr = weight_data.unsafe_ptr().bitcast[Float32]()

        # Check if biases are present (file size indicates this)
        var has_bias = 0
        if len(weight_data) == total_weight_size * 4:
            has_bias = 1
            print("Model has QKV biases")
        f.close()

        # Upload weights to GPU
        var upload_size = base_weights * 4
        if has_bias == 1:
            upload_size = total_weight_size * 4
        ret = self.handle.call["cublas_upload_weights", Int32](
            weight_ptr, Int64(upload_size)
        )
        if ret != 0:
            print("Failed to upload weights")
            return False

        # Configure GPU with model dimensions
        ret = self.handle.call["gpu_configure", Int32](
            Int32(d), Int32(hd), Int32(n), Int32(config.n_heads),
            Int32(config.n_kv_heads), Int32(v), Int32(s), Int32(has_bias)
        )
        if ret != 0:
            print("Failed to configure GPU")
            return False

        self.configured = True
        print("GPU initialized and configured")
        return True

    fn forward(self, token: Int, pos: Int) -> Int:
        """Forward pass. Returns next token."""
        if not self.configured:
            return -1
        return Int(self.handle.call["gpu_forward", Int32](Int32(token), Int32(pos)))

    fn cleanup(self):
        if self.available:
            self.handle.call["cublas_cleanup", NoneType]()


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
    var checkpoint = "models/qwen2.5-1.5b.bin"
    var tokenizer_path = "models/qwen2.5-1.5b_tokenizer.bin"
    var steps = 100
    var prompt = String("Hello")
    var lib_path = "./lib/libcublas_matmul.so"
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
    print("EdgeLLM GPU Inference - Full CUDA Acceleration")
    print("=" * 60)
    print()

    # Load config
    print("Loading model config from", checkpoint)
    var config = Config(checkpoint)

    print("Config:")
    print("  dim =", config.dim)
    print("  hidden_dim =", config.hidden_dim)
    print("  n_layers =", config.n_layers)
    print("  n_heads =", config.n_heads, "n_kv_heads =", config.n_kv_heads)
    print("  vocab_size =", config.vocab_size)
    print("  seq_len =", config.seq_len)
    print()

    # Initialize GPU
    print("Initializing GPU...")
    var gpu = GPUInference(lib_path)

    if not gpu.available:
        print("ERROR: GPU library not available at", lib_path)
        print("Build with: make -C src/kernels/cuda cublas")
        return

    # Initialize and upload weights
    print("Uploading weights to GPU...")
    if not gpu.init_and_upload(checkpoint, config):
        print("ERROR: Failed to initialize GPU")
        return

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
