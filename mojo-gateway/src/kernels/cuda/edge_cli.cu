/**
 * edge - Minimal LLM inference CLI
 *
 * Usage:
 *   edge run <model>              Interactive chat
 *   edge run <model> -p "prompt"  Single generation
 *   edge models                   List available models
 *
 * Example:
 *   edge run qwen
 *   edge run qwen -p "What is 2+2?"
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>

// ============================================================================
// External CUDA functions
// ============================================================================

extern "C" {
    int cublas_init_int4(float*, int, int, int, int, int, int, int, int, int);
    int cublas_upload_int4_weights(const uint8_t*, const half*, size_t, size_t);
    void gpu_configure_int4(int, int, int, int, int, int, int, int, int, int);
    int gpu_forward_int4(int token, int pos);
}

// ============================================================================
// Model registry (hardcoded paths - edit for your setup)
// ============================================================================

struct Model {
    const char* name;
    const char* path;
    const char* tokenizer;
    const char* template_user;
    const char* template_asst;
};

static Model MODELS[] = {
    {
        "qwen",
        "models/qwen2.5-1.5b_int4.bin",
        "models/qwen2.5-1.5b_int4_tokenizer.bin",
        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
        "<|im_end|>\n"
    },
    {
        "llama",
        "models/llama-3.2-1b_int4.bin",
        "models/llama-3.2-1b_int4_tokenizer.bin",
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        "<|eot_id|>"
    },
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

// ============================================================================
// Model config
// ============================================================================

struct Config {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads;
    int vocab_size, seq_len, head_dim, kv_dim;
};

// ============================================================================
// Tokenizer (llama.c format)
// ============================================================================

static int* vocab_offsets = nullptr;
static char* vocab_data = nullptr;
static int vocab_size = 0;
static float* vocab_scores = nullptr;

int load_tokenizer(const char* path, int expected_vocab) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open tokenizer %s\n", path); return -1; }

    int max_len;
    fread(&max_len, 4, 1, f);

    vocab_size = expected_vocab;
    vocab_offsets = (int*)malloc((vocab_size + 1) * sizeof(int));
    vocab_scores = (float*)malloc(vocab_size * sizeof(float));

    // First pass: calculate total size
    long start = ftell(f);
    size_t total = 0;
    for (int i = 0; i < vocab_size; i++) {
        float score; int len;
        fread(&score, 4, 1, f);
        fread(&len, 4, 1, f);
        fseek(f, len, SEEK_CUR);
        total += len + 1;
    }

    vocab_data = (char*)malloc(total);
    fseek(f, start, SEEK_SET);

    // Second pass: load tokens
    size_t offset = 0;
    for (int i = 0; i < vocab_size; i++) {
        float score; int len;
        fread(&score, 4, 1, f);
        fread(&len, 4, 1, f);
        vocab_scores[i] = score;
        vocab_offsets[i] = offset;
        fread(vocab_data + offset, 1, len, f);
        vocab_data[offset + len] = '\0';
        offset += len + 1;
    }
    vocab_offsets[vocab_size] = offset;

    fclose(f);
    return 0;
}

const char* decode_token(int id) {
    if (id < 0 || id >= vocab_size) return "";
    return vocab_data + vocab_offsets[id];
}

int encode_single(const char* text, int* tokens, int max_tokens) {
    // Greedy BPE encoding
    int n = 0;
    size_t len = strlen(text);
    size_t i = 0;

    while (i < len && n < max_tokens) {
        int best_id = -1;
        int best_len = 0;

        for (int id = 0; id < vocab_size; id++) {
            const char* tok = vocab_data + vocab_offsets[id];
            int tok_len = vocab_offsets[id + 1] - vocab_offsets[id] - 1;
            if (tok_len > 0 && tok_len > best_len && i + tok_len <= len) {
                if (memcmp(text + i, tok, tok_len) == 0) {
                    best_id = id;
                    best_len = tok_len;
                }
            }
        }

        if (best_id >= 0) {
            tokens[n++] = best_id;
            i += best_len;
        } else {
            i++; // skip unknown byte
        }
    }
    return n;
}

// ============================================================================
// Model loading
// ============================================================================

static uint8_t* weights_int4 = nullptr;
static half* scales = nullptr;
static float* weights_fp32 = nullptr;
static size_t g_fp32_size = 0;
static size_t g_int4_bytes = 0;
static size_t g_scales_bytes = 0;

int load_model(const char* path, Config* cfg) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open model %s\n", path); return -1; }

    // Read 8-field header: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, is_int4
    int header[8];
    if (fread(header, 4, 8, f) != 8) {
        fprintf(stderr, "error: failed to read model header\n");
        fclose(f);
        return -1;
    }

    cfg->dim = header[0];
    cfg->hidden_dim = header[1];
    cfg->n_layers = header[2];
    cfg->n_heads = header[3];
    cfg->n_kv_heads = header[4];
    cfg->vocab_size = header[5];
    cfg->seq_len = header[6];
    cfg->head_dim = cfg->dim / cfg->n_heads;
    cfg->kv_dim = (cfg->dim * cfg->n_kv_heads) / cfg->n_heads;

    int L = cfg->n_layers, D = cfg->dim, H = cfg->hidden_dim;
    int V = cfg->vocab_size, KV = cfg->kv_dim, S = cfg->seq_len;
    int head_dim = cfg->head_dim;
    int group_size = 128;

    // Calculate FP32 size (must match export script exactly)
    size_t fp32_size = 0;
    fp32_size += (size_t)V * D;                     // embeddings
    fp32_size += (size_t)L * D;                     // rms_att (stacked)
    fp32_size += (size_t)L * D;                     // rms_ffn (stacked)
    fp32_size += D;                                 // rms_final
    fp32_size += (size_t)S * (head_dim / 2);        // freq_cos
    fp32_size += (size_t)S * (head_dim / 2);        // freq_sin
    fp32_size += (size_t)L * D;                     // bq
    fp32_size += (size_t)L * KV;                    // bk
    fp32_size += (size_t)L * KV;                    // bv

    // Calculate INT4 packed size
    size_t int4_elements = 0;
    for (int l = 0; l < L; l++) {
        int4_elements += (size_t)D * D;             // wq
        int4_elements += (size_t)KV * D;            // wk
        int4_elements += (size_t)KV * D;            // wv
        int4_elements += (size_t)D * D;             // wo
        int4_elements += (size_t)H * D;             // w1
        int4_elements += (size_t)D * H;             // w2
        int4_elements += (size_t)H * D;             // w3
    }
    size_t int4_bytes = int4_elements / 2;

    // Calculate scales size
    int groups_dim = (D + group_size - 1) / group_size;
    int groups_hd = (H + group_size - 1) / group_size;
    size_t n_scales = 0;
    for (int l = 0; l < L; l++) {
        n_scales += (size_t)D * groups_dim;         // wq
        n_scales += (size_t)KV * groups_dim;        // wk
        n_scales += (size_t)KV * groups_dim;        // wv
        n_scales += (size_t)D * groups_dim;         // wo
        n_scales += (size_t)H * groups_dim;         // w1
        n_scales += (size_t)D * groups_hd;          // w2
        n_scales += (size_t)H * groups_dim;         // w3
    }
    size_t scales_bytes = n_scales * sizeof(half);

    // Save sizes for init_gpu
    g_fp32_size = fp32_size;
    g_int4_bytes = int4_bytes;
    g_scales_bytes = scales_bytes;

    // Allocate
    weights_fp32 = (float*)malloc(fp32_size * sizeof(float));
    scales = (half*)malloc(scales_bytes);
    weights_int4 = (uint8_t*)malloc(int4_bytes);

    if (!weights_fp32 || !scales || !weights_int4) {
        fprintf(stderr, "error: failed to allocate memory\n");
        fclose(f);
        return -1;
    }

    // Model file format: header -> FP32 -> scales(FP16) -> packed
    if (fread(weights_fp32, sizeof(float), fp32_size, f) != fp32_size) {
        fprintf(stderr, "error: failed to read FP32 weights\n");
        fclose(f);
        return -1;
    }
    if (fread(scales, 1, scales_bytes, f) != scales_bytes) {
        fprintf(stderr, "error: failed to read scales\n");
        fclose(f);
        return -1;
    }
    if (fread(weights_int4, 1, int4_bytes, f) != int4_bytes) {
        fprintf(stderr, "error: failed to read INT4 weights\n");
        fclose(f);
        return -1;
    }

    fclose(f);

    printf("  %d layers, %d dim, %d vocab\n", L, D, V);
    return 0;
}

// ============================================================================
// GPU initialization
// ============================================================================

int init_gpu(Config* cfg) {
    cublas_init_int4(weights_fp32, cfg->dim, cfg->hidden_dim, cfg->n_layers,
                     cfg->n_heads, cfg->n_kv_heads, cfg->vocab_size, cfg->seq_len,
                     cfg->head_dim, cfg->kv_dim);

    cublas_upload_int4_weights(weights_int4, scales, g_int4_bytes, g_scales_bytes);

    gpu_configure_int4(cfg->dim, cfg->hidden_dim, cfg->n_layers,
                       cfg->n_heads, cfg->n_kv_heads, cfg->vocab_size,
                       cfg->seq_len, cfg->head_dim, cfg->kv_dim, 128);

    return 0;
}

// ============================================================================
// Generation
// ============================================================================

void generate(const char* prompt, int max_tokens, Config* cfg, Model* model) {
    // Skip chat template for now - just use raw prompt
    // TODO: Add proper special token encoding for chat templates

    // Encode
    int tokens[2048];
    int n_tokens = encode_single(prompt, tokens, 2048);

    if (n_tokens == 0) {
        fprintf(stderr, "error: failed to encode prompt\n");
        return;
    }

    printf("\n");

    // Prefill: process all tokens except the last
    for (int i = 0; i < n_tokens - 1; i++) {
        gpu_forward_int4(tokens[i], i);
    }

    // Generate
    int token = tokens[n_tokens - 1];
    int pos = n_tokens - 1;
    int generated = 0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    while (generated < max_tokens) {
        // Greedy decoding
        token = gpu_forward_int4(token, pos);
        pos++;
        generated++;

        // Decode and print
        const char* text = decode_token(token);

        // Check for EOS tokens
        if (token == 151643 || token == 151645 || token == 2 ||
            strstr(text, "<|im_end|>") || strstr(text, "<|eot_id|>")) {
            break;
        }

        printf("%s", text);
        fflush(stdout);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, start, end);

    printf("\n\n[%d tokens, %.1f tok/s]\n", generated, generated * 1000.0f / ms);
}

// ============================================================================
// Commands
// ============================================================================

void cmd_models() {
    printf("\nAvailable models:\n");
    printf("  %-10s %s\n", "NAME", "PATH");
    printf("  %-10s %s\n", "----", "----");
    for (int i = 0; MODELS[i].name; i++) {
        const char* status = access(MODELS[i].path, F_OK) == 0 ? "[ready]" : "[not found]";
        printf("  %-10s %s %s\n", MODELS[i].name, MODELS[i].path, status);
    }
    printf("\nUsage: edge run <model>\n\n");
}

Model* find_model(const char* name) {
    for (int i = 0; MODELS[i].name; i++) {
        if (strcmp(MODELS[i].name, name) == 0) return &MODELS[i];
    }
    return nullptr;
}

void cmd_run(const char* model_name, const char* prompt, int max_tokens) {
    Model* model = find_model(model_name);
    if (!model) {
        fprintf(stderr, "error: unknown model '%s'\n", model_name);
        cmd_models();
        return;
    }

    // Check files exist
    if (access(model->path, F_OK) != 0) {
        fprintf(stderr, "error: model file not found: %s\n", model->path);
        return;
    }
    if (access(model->tokenizer, F_OK) != 0) {
        fprintf(stderr, "error: tokenizer not found: %s\n", model->tokenizer);
        return;
    }

    printf("Loading %s...\n", model->name);

    Config cfg;
    if (load_model(model->path, &cfg) != 0) return;
    if (load_tokenizer(model->tokenizer, cfg.vocab_size) != 0) return;
    if (init_gpu(&cfg) != 0) return;

    printf("Ready.\n");

    if (prompt) {
        // Single generation
        generate(prompt, max_tokens, &cfg, model);
    } else {
        // Interactive mode
        printf("\nType your message (or 'exit' to quit):\n");
        char input[4096];
        while (1) {
            printf("\n> ");
            fflush(stdout);
            if (!fgets(input, sizeof(input), stdin)) break;

            // Remove newline
            input[strcspn(input, "\n")] = 0;

            if (strcmp(input, "exit") == 0 || strcmp(input, "quit") == 0) break;
            if (strlen(input) == 0) continue;

            generate(input, max_tokens, &cfg, model);
        }
        printf("\nBye.\n");
    }
}

void print_help() {
    printf("\nedge - Fast LLM inference\n\n");
    printf("Usage:\n");
    printf("  edge run <model>              Interactive chat\n");
    printf("  edge run <model> -p \"prompt\"  Single generation\n");
    printf("  edge models                   List available models\n");
    printf("\nOptions:\n");
    printf("  -n <tokens>   Max tokens to generate (default: 256)\n");
    printf("  -t <temp>     Temperature (default: 0.7)\n");
    printf("  -p <prompt>   Single prompt (non-interactive)\n");
    printf("\nExamples:\n");
    printf("  edge run qwen\n");
    printf("  edge run qwen -p \"What is 2+2?\"\n\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help();
        return 0;
    }

    const char* cmd = argv[1];

    if (strcmp(cmd, "models") == 0) {
        cmd_models();
        return 0;
    }

    if (strcmp(cmd, "run") == 0) {
        if (argc < 3) {
            fprintf(stderr, "error: missing model name\n");
            print_help();
            return 1;
        }

        const char* model = argv[2];
        const char* prompt = nullptr;
        int max_tokens = 256;
        float temperature = 0.7f;

        // Parse options
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
                prompt = argv[++i];
            } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                max_tokens = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                temperature = atof(argv[++i]);
            }
        }

        cmd_run(model, prompt, max_tokens);
        return 0;
    }

    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "-h") == 0 || strcmp(cmd, "--help") == 0) {
        print_help();
        return 0;
    }

    fprintf(stderr, "error: unknown command '%s'\n", cmd);
    print_help();
    return 1;
}
