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
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <signal.h>

// ============================================================================
// External CUDA functions
// ============================================================================

extern "C" {
    int cublas_init_int4(float*, int, int, int, int, int, int, int, int, int);
    int cublas_upload_int4_weights(const uint8_t*, const half*, size_t, size_t);
    void gpu_configure_int4(int, int, int, int, int, int, int, int, int, int);
    int gpu_forward_int4(int token, int pos);

    // Sampling functions (from inference_with_sampling.cu)
    int sampling_config_init(int vocab_size, float temperature, int top_k, float top_p, float repetition_penalty);
    void set_sampling_params(float temperature, int top_k, float top_p, float rep_penalty);
    int forward_with_sampling(int token, int pos);
    void reset_past_tokens();
    void sampling_cleanup_resources();
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
// Sampling config
// ============================================================================

struct SamplingParams {
    float temperature;
    int top_k;
    float top_p;
    float repetition_penalty;
};

static SamplingParams g_sampling = {
    .temperature = 0.7f,
    .top_k = 40,
    .top_p = 0.9f,
    .repetition_penalty = 1.1f
};

static bool g_sampling_initialized = false;

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

    // Initialize sampling if needed
    if (!g_sampling_initialized && g_sampling.temperature > 0.0f) {
        sampling_config_init(cfg->vocab_size, g_sampling.temperature,
                            g_sampling.top_k, g_sampling.top_p,
                            g_sampling.repetition_penalty);
        g_sampling_initialized = true;
    } else if (g_sampling_initialized) {
        // Update params and reset for new generation
        set_sampling_params(g_sampling.temperature, g_sampling.top_k,
                           g_sampling.top_p, g_sampling.repetition_penalty);
        reset_past_tokens();
    }

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
        // Use sampling if temperature > 0, otherwise greedy
        if (g_sampling.temperature > 0.0f) {
            token = forward_with_sampling(token, pos);
        } else {
            token = gpu_forward_int4(token, pos);
        }
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

    printf("\n\n[%d tokens, %.1f tok/s, temp=%.2f]\n", generated, generated * 1000.0f / ms, g_sampling.temperature);
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
    printf("  edge serve <model> [port]     Start HTTP API server\n");
    printf("\nOptions:\n");
    printf("  -n <tokens>   Max tokens to generate (default: 256)\n");
    printf("  -t <temp>     Temperature (default: 0.7, 0=greedy)\n");
    printf("  -k <top_k>    Top-k sampling (default: 40, 0=disabled)\n");
    printf("  --top_p <p>   Top-p nucleus sampling (default: 0.9)\n");
    printf("  --rep_pen <p> Repetition penalty (default: 1.1)\n");
    printf("  -p <prompt>   Single prompt (non-interactive)\n");
    printf("\nExamples:\n");
    printf("  edge run qwen\n");
    printf("  edge run qwen -p \"What is 2+2?\"\n");
    printf("  edge run qwen -t 0.9 -k 50 -p \"Write a poem\"\n");
    printf("  edge run qwen -t 0                          # Greedy\n");
    printf("  edge serve qwen 8080                        # HTTP server\n\n");
}

// ============================================================================
// HTTP Server for Ollama-compatible API
// ============================================================================

static volatile bool g_server_running = true;
static Config* g_server_cfg = nullptr;
static Model* g_server_model = nullptr;
static pthread_mutex_t g_inference_mutex = PTHREAD_MUTEX_INITIALIZER;

void handle_sigint(int sig) {
    (void)sig;
    g_server_running = false;
    printf("\nShutting down...\n");
}

// Simple JSON parsing helpers
static char* json_get_string(const char* json, const char* key, char* buf, size_t bufsize) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* start = strstr(json, pattern);
    if (!start) return nullptr;
    start += strlen(pattern);
    while (*start == ' ' || *start == '\t') start++;
    if (*start != '"') return nullptr;
    start++;
    const char* end = strchr(start, '"');
    if (!end) return nullptr;
    size_t len = end - start;
    if (len >= bufsize) len = bufsize - 1;
    memcpy(buf, start, len);
    buf[len] = '\0';
    return buf;
}

static float json_get_float(const char* json, const char* key, float default_val) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* start = strstr(json, pattern);
    if (!start) return default_val;
    start += strlen(pattern);
    while (*start == ' ' || *start == '\t') start++;
    return (float)atof(start);
}

static int json_get_int(const char* json, const char* key, int default_val) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* start = strstr(json, pattern);
    if (!start) return default_val;
    start += strlen(pattern);
    while (*start == ' ' || *start == '\t') start++;
    return atoi(start);
}

static bool json_get_bool(const char* json, const char* key, bool default_val) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* start = strstr(json, pattern);
    if (!start) return default_val;
    start += strlen(pattern);
    while (*start == ' ' || *start == '\t') start++;
    if (strncmp(start, "true", 4) == 0) return true;
    if (strncmp(start, "false", 5) == 0) return false;
    return default_val;
}

// Send HTTP response headers
static void send_headers(int client_fd, int status, const char* content_type, bool streaming) {
    char response[512];
    const char* status_text = (status == 200) ? "OK" : "Bad Request";
    if (streaming) {
        snprintf(response, sizeof(response),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "\r\n", status, status_text, content_type);
    } else {
        snprintf(response, sizeof(response),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Connection: close\r\n"
            "\r\n", status, status_text, content_type);
    }
    send(client_fd, response, strlen(response), 0);
}

// Send SSE chunk
static void send_chunk(int client_fd, const char* data) {
    char chunk[1024];
    int len = snprintf(chunk, sizeof(chunk), "%lx\r\n%s\r\n", strlen(data), data);
    send(client_fd, chunk, len, 0);
}

// Generate with streaming output
static void generate_streaming(int client_fd, const char* prompt, int max_tokens, bool stream) {
    pthread_mutex_lock(&g_inference_mutex);

    // Encode prompt
    int tokens[2048];
    int n_tokens = encode_single(prompt, tokens, 2048);
    if (n_tokens == 0) {
        pthread_mutex_unlock(&g_inference_mutex);
        const char* error = "{\"error\":\"failed to encode prompt\"}";
        if (stream) send_chunk(client_fd, error);
        else send(client_fd, error, strlen(error), 0);
        return;
    }

    // Initialize sampling if needed
    if (!g_sampling_initialized && g_sampling.temperature > 0.0f) {
        sampling_config_init(g_server_cfg->vocab_size, g_sampling.temperature,
                            g_sampling.top_k, g_sampling.top_p,
                            g_sampling.repetition_penalty);
        g_sampling_initialized = true;
    } else if (g_sampling_initialized) {
        set_sampling_params(g_sampling.temperature, g_sampling.top_k,
                           g_sampling.top_p, g_sampling.repetition_penalty);
        reset_past_tokens();
    }

    // Prefill
    for (int i = 0; i < n_tokens - 1; i++) {
        gpu_forward_int4(tokens[i], i);
    }

    // Generate
    int token = tokens[n_tokens - 1];
    int pos = n_tokens - 1;
    int generated = 0;
    char response_buf[65536] = "";
    size_t response_len = 0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    while (generated < max_tokens && g_server_running) {
        if (g_sampling.temperature > 0.0f) {
            token = forward_with_sampling(token, pos);
        } else {
            token = gpu_forward_int4(token, pos);
        }
        pos++;
        generated++;

        const char* text = decode_token(token);

        // Check for EOS
        if (token == 151643 || token == 151645 || token == 2 ||
            strstr(text, "<|im_end|>") || strstr(text, "<|eot_id|>")) {
            break;
        }

        if (stream) {
            // Send streaming response (Ollama format)
            char json[512];
            // Escape special characters in text
            char escaped[256];
            size_t j = 0;
            for (size_t i = 0; text[i] && j < sizeof(escaped) - 2; i++) {
                if (text[i] == '"') { escaped[j++] = '\\'; escaped[j++] = '"'; }
                else if (text[i] == '\\') { escaped[j++] = '\\'; escaped[j++] = '\\'; }
                else if (text[i] == '\n') { escaped[j++] = '\\'; escaped[j++] = 'n'; }
                else if (text[i] == '\r') { escaped[j++] = '\\'; escaped[j++] = 'r'; }
                else if (text[i] == '\t') { escaped[j++] = '\\'; escaped[j++] = 't'; }
                else escaped[j++] = text[i];
            }
            escaped[j] = '\0';

            snprintf(json, sizeof(json),
                "{\"model\":\"%s\",\"response\":\"%s\",\"done\":false}",
                g_server_model->name, escaped);
            send_chunk(client_fd, json);
            send_chunk(client_fd, "\n");
        } else {
            // Accumulate for non-streaming response
            size_t text_len = strlen(text);
            if (response_len + text_len < sizeof(response_buf) - 1) {
                memcpy(response_buf + response_len, text, text_len);
                response_len += text_len;
            }
        }
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, start, end);

    if (stream) {
        // Send final chunk
        char final_json[256];
        snprintf(final_json, sizeof(final_json),
            "{\"model\":\"%s\",\"response\":\"\",\"done\":true,"
            "\"total_duration\":%d,\"eval_count\":%d,\"eval_duration\":%d}",
            g_server_model->name, (int)(ms * 1000000), generated, (int)(ms * 1000000));
        send_chunk(client_fd, final_json);
        send_chunk(client_fd, "\n");
        send(client_fd, "0\r\n\r\n", 5, 0);  // End chunked transfer
    } else {
        response_buf[response_len] = '\0';
        // Escape for JSON
        char escaped_response[65536];
        size_t j = 0;
        for (size_t i = 0; response_buf[i] && j < sizeof(escaped_response) - 2; i++) {
            if (response_buf[i] == '"') { escaped_response[j++] = '\\'; escaped_response[j++] = '"'; }
            else if (response_buf[i] == '\\') { escaped_response[j++] = '\\'; escaped_response[j++] = '\\'; }
            else if (response_buf[i] == '\n') { escaped_response[j++] = '\\'; escaped_response[j++] = 'n'; }
            else if (response_buf[i] == '\r') { escaped_response[j++] = '\\'; escaped_response[j++] = 'r'; }
            else if (response_buf[i] == '\t') { escaped_response[j++] = '\\'; escaped_response[j++] = 't'; }
            else escaped_response[j++] = response_buf[i];
        }
        escaped_response[j] = '\0';

        char json[70000];
        snprintf(json, sizeof(json),
            "{\"model\":\"%s\",\"response\":\"%s\",\"done\":true,"
            "\"total_duration\":%d,\"eval_count\":%d,\"eval_duration\":%d}",
            g_server_model->name, escaped_response, (int)(ms * 1000000), generated, (int)(ms * 1000000));
        send(client_fd, json, strlen(json), 0);
    }

    pthread_mutex_unlock(&g_inference_mutex);
}

// Handle HTTP request
static void handle_request(int client_fd) {
    char buffer[65536];
    ssize_t bytes = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (bytes <= 0) {
        close(client_fd);
        return;
    }
    buffer[bytes] = '\0';

    // Parse method and path
    char method[16] = "", path[256] = "";
    sscanf(buffer, "%15s %255s", method, path);

    // Find body (after \r\n\r\n)
    const char* body = strstr(buffer, "\r\n\r\n");
    if (body) body += 4;
    else body = "";

    // Route: GET /api/tags - List models
    if (strcmp(method, "GET") == 0 && strcmp(path, "/api/tags") == 0) {
        send_headers(client_fd, 200, "application/json", false);
        char json[1024] = "{\"models\":[";
        for (int i = 0; MODELS[i].name; i++) {
            if (i > 0) strcat(json, ",");
            char model_json[256];
            snprintf(model_json, sizeof(model_json),
                "{\"name\":\"%s\",\"size\":0,\"digest\":\"\",\"modified_at\":\"\"}",
                MODELS[i].name);
            strcat(json, model_json);
        }
        strcat(json, "]}");
        send(client_fd, json, strlen(json), 0);
    }
    // Route: POST /api/generate - Generate text
    else if (strcmp(method, "POST") == 0 && strcmp(path, "/api/generate") == 0) {
        char prompt[8192];
        if (!json_get_string(body, "prompt", prompt, sizeof(prompt))) {
            send_headers(client_fd, 400, "application/json", false);
            const char* error = "{\"error\":\"missing prompt\"}";
            send(client_fd, error, strlen(error), 0);
            close(client_fd);
            return;
        }

        // Parse options
        g_sampling.temperature = json_get_float(body, "temperature", g_sampling.temperature);
        g_sampling.top_k = json_get_int(body, "top_k", g_sampling.top_k);
        g_sampling.top_p = json_get_float(body, "top_p", g_sampling.top_p);
        g_sampling.repetition_penalty = json_get_float(body, "repeat_penalty", g_sampling.repetition_penalty);
        int max_tokens = json_get_int(body, "num_predict", 256);
        bool stream = json_get_bool(body, "stream", true);

        send_headers(client_fd, 200, "application/x-ndjson", stream);
        generate_streaming(client_fd, prompt, max_tokens, stream);
    }
    // Route: POST /api/chat - Chat completion (simplified)
    else if (strcmp(method, "POST") == 0 && strcmp(path, "/api/chat") == 0) {
        // Extract last message content (simplified - just gets prompt field)
        char prompt[8192];
        const char* messages = strstr(body, "\"messages\"");
        if (messages) {
            const char* content = strstr(messages, "\"content\":");
            if (content) {
                content += 10;
                while (*content == ' ' || *content == '"') content++;
                const char* end = strchr(content, '"');
                if (end) {
                    size_t len = end - content;
                    if (len >= sizeof(prompt)) len = sizeof(prompt) - 1;
                    memcpy(prompt, content, len);
                    prompt[len] = '\0';
                } else {
                    strcpy(prompt, "Hello");
                }
            } else {
                strcpy(prompt, "Hello");
            }
        } else if (!json_get_string(body, "prompt", prompt, sizeof(prompt))) {
            strcpy(prompt, "Hello");
        }

        g_sampling.temperature = json_get_float(body, "temperature", g_sampling.temperature);
        int max_tokens = json_get_int(body, "num_predict", 256);
        bool stream = json_get_bool(body, "stream", true);

        send_headers(client_fd, 200, "application/x-ndjson", stream);
        generate_streaming(client_fd, prompt, max_tokens, stream);
    }
    // Health check
    else if (strcmp(method, "GET") == 0 && (strcmp(path, "/") == 0 || strcmp(path, "/health") == 0)) {
        send_headers(client_fd, 200, "application/json", false);
        const char* json = "{\"status\":\"ok\"}";
        send(client_fd, json, strlen(json), 0);
    }
    // 404
    else {
        send_headers(client_fd, 404, "application/json", false);
        const char* error = "{\"error\":\"not found\"}";
        send(client_fd, error, strlen(error), 0);
    }

    close(client_fd);
}

void cmd_serve(const char* model_name, int port) {
    Model* model = find_model(model_name);
    if (!model) {
        fprintf(stderr, "error: unknown model '%s'\n", model_name);
        cmd_models();
        return;
    }

    if (access(model->path, F_OK) != 0) {
        fprintf(stderr, "error: model file not found: %s\n", model->path);
        return;
    }
    if (access(model->tokenizer, F_OK) != 0) {
        fprintf(stderr, "error: tokenizer not found: %s\n", model->tokenizer);
        return;
    }

    printf("Loading %s...\n", model->name);

    static Config cfg;
    if (load_model(model->path, &cfg) != 0) return;
    if (load_tokenizer(model->tokenizer, cfg.vocab_size) != 0) return;
    if (init_gpu(&cfg) != 0) return;

    g_server_cfg = &cfg;
    g_server_model = model;

    // Create socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        fprintf(stderr, "error: failed to create socket\n");
        return;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "error: failed to bind to port %d\n", port);
        close(server_fd);
        return;
    }

    if (listen(server_fd, 10) < 0) {
        fprintf(stderr, "error: failed to listen\n");
        close(server_fd);
        return;
    }

    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    printf("\n");
    printf("EdgeLLM server running at http://localhost:%d\n", port);
    printf("Model: %s (temp=%.2f, top_k=%d, top_p=%.2f)\n",
           model->name, g_sampling.temperature, g_sampling.top_k, g_sampling.top_p);
    printf("\nEndpoints:\n");
    printf("  GET  /api/tags      List available models\n");
    printf("  POST /api/generate  Generate text\n");
    printf("  POST /api/chat      Chat completion\n");
    printf("  GET  /health        Health check\n");
    printf("\nPress Ctrl+C to stop.\n\n");

    while (g_server_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (g_server_running) {
                fprintf(stderr, "warning: accept failed\n");
            }
            continue;
        }

        handle_request(client_fd);
    }

    close(server_fd);
    printf("Server stopped.\n");
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

        // Parse options
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
                prompt = argv[++i];
            } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                max_tokens = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                g_sampling.temperature = atof(argv[++i]);
            } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
                g_sampling.top_k = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--top_p") == 0 && i + 1 < argc) {
                g_sampling.top_p = atof(argv[++i]);
            } else if (strcmp(argv[i], "--rep_pen") == 0 && i + 1 < argc) {
                g_sampling.repetition_penalty = atof(argv[++i]);
            }
        }

        cmd_run(model, prompt, max_tokens);
        return 0;
    }

    if (strcmp(cmd, "serve") == 0) {
        if (argc < 3) {
            fprintf(stderr, "error: missing model name\n");
            print_help();
            return 1;
        }

        const char* model = argv[2];
        int port = 11434;  // Default Ollama port

        // Parse options
        for (int i = 3; i < argc; i++) {
            if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0) && i + 1 < argc) {
                port = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                g_sampling.temperature = atof(argv[++i]);
            } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
                g_sampling.top_k = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--top_p") == 0 && i + 1 < argc) {
                g_sampling.top_p = atof(argv[++i]);
            } else if (strcmp(argv[i], "--rep_pen") == 0 && i + 1 < argc) {
                g_sampling.repetition_penalty = atof(argv[++i]);
            } else if (argv[i][0] != '-') {
                // Positional argument - could be port
                port = atoi(argv[i]);
            }
        }

        cmd_serve(model, port);
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
