# FlashAttention-2 Optimized Architecture

## System Overview

```mermaid
flowchart TB
    subgraph HOST["HOST (CPU)"]
        INPUT["Input Tokens"]
        OUTPUT["Generated Tokens"]
        TOKENIZER["Tokenizer"]
    end

    subgraph GPU["GPU (CUDA)"]
        subgraph PERSISTENT["Persistent GPU Memory (Point 2)"]
            Q_BUF["Q Buffer<br/>[batch_heads × head_dim]<br/>FP16"]
            KV_CACHE["KV Cache<br/>[batch_heads × max_seq × head_dim]<br/>FP16"]
            O_BUF["Output Buffer<br/>[batch_heads × head_dim]<br/>FP32"]
            PARTIAL["Split-K Buffers<br/>partial_out + partial_lse"]
        end

        subgraph KERNEL["Optimized FA2 Kernel"]
            subgraph WMMA["Point 1: Tensor Cores"]
                FP16_COMPUTE["FP16 Compute<br/>half2 vectorized"]
                SCORE_CALC["Q @ K^T<br/>(WMMA ready)"]
            end

            subgraph SPLITK["Point 3: FlashDecoding++"]
                SPLIT["Split KV Cache<br/>into K chunks"]
                PARALLEL["Parallel Attention<br/>per split"]
                REDUCE["Reduce via<br/>log-sum-exp"]
            end

            SOFTMAX["Online Softmax<br/>(unified max)"]
            PV_MULT["P @ V<br/>Output accumulation"]
        end
    end

    INPUT --> TOKENIZER
    TOKENIZER -->|"Once per token"| Q_BUF
    TOKENIZER -->|"Update cache"| KV_CACHE

    Q_BUF --> FP16_COMPUTE
    KV_CACHE --> SPLIT
    SPLIT --> PARALLEL
    FP16_COMPUTE --> SCORE_CALC
    SCORE_CALC --> SOFTMAX
    PARALLEL --> SOFTMAX
    SOFTMAX --> PV_MULT
    PV_MULT --> REDUCE
    REDUCE --> O_BUF

    O_BUF -->|"Once at end"| OUTPUT

    style PERSISTENT fill:#e1f5fe
    style WMMA fill:#fff3e0
    style SPLITK fill:#e8f5e9
```

## Detailed Processing Flow

```mermaid
sequenceDiagram
    participant Host as Host (CPU)
    participant Stream1 as Copy Stream
    participant Stream2 as Compute Stream
    participant GPU as GPU Memory

    Note over Host,GPU: INITIALIZATION (Once)
    Host->>GPU: flash_attention_v2_opt_init()
    GPU->>GPU: Allocate Q, KV Cache, O buffers
    GPU->>GPU: Create CUDA streams

    Note over Host,GPU: TOKEN GENERATION LOOP
    loop For each token
        Host->>Stream1: flash_attention_v2_opt_load_qkv()
        Stream1->>GPU: Async copy Q (FP32→FP16)
        Stream1->>GPU: Update KV cache at position

        Host->>Stream2: flash_attention_v2_opt_decode_gpu()

        alt seq_len > 256
            Stream2->>GPU: Launch Split-K kernel
            Stream2->>GPU: Launch Reduce kernel
        else seq_len <= 256
            Stream2->>GPU: Launch WMMA kernel
        end

        Stream2->>Stream2: Sync compute stream
    end

    Note over Host,GPU: GET OUTPUT (Once)
    Host->>GPU: flash_attention_v2_opt_get_output()
    GPU->>Host: Copy O buffer (FP32)
```

## Memory Layout

```mermaid
block-beta
    columns 3

    block:HOST_MEM:1
        columns 1
        H1["Input Q (FP32)"]
        H2["Input K (FP32)"]
        H3["Input V (FP32)"]
        H4["Output O (FP32)"]
    end

    block:TRANSFER:1
        columns 1
        T1["cudaMemcpyAsync"]
        T2["FP32 → FP16"]
        T3["Convert"]
        T4["cudaMemcpyAsync"]
    end

    block:GPU_MEM:1
        columns 1
        G1["d_Q (FP16)<br/>batch_heads × head_dim"]
        G2["d_K_cache (FP16)<br/>batch_heads × max_seq × head_dim"]
        G3["d_V_cache (FP16)<br/>batch_heads × max_seq × head_dim"]
        G4["d_O (FP32)<br/>batch_heads × head_dim"]
    end

    H1 --> T1 --> G1
    H2 --> T2 --> G2
    H3 --> T3 --> G3
    G4 --> T4 --> H4

    style HOST_MEM fill:#ffebee
    style TRANSFER fill:#fff8e1
    style GPU_MEM fill:#e8f5e9
```

## Kernel Architecture

```mermaid
flowchart LR
    subgraph INPUT["Input (per head)"]
        Q["Q[1, head_dim]"]
        K["K[seq_len, head_dim]"]
        V["V[seq_len, head_dim]"]
    end

    subgraph TILING["Tiled Processing"]
        direction TB
        TILE1["Tile 0<br/>[0:64]"]
        TILE2["Tile 1<br/>[64:128]"]
        TILE3["Tile 2<br/>[128:192]"]
        TILEN["Tile N<br/>[...]"]
    end

    subgraph ATTENTION["Per-Tile Attention"]
        direction TB
        LOAD["Load K,V tile<br/>to shared mem"]
        DOT["Q @ K^T<br/>(FP16 vectorized)"]
        SCALE["Scale by<br/>1/√head_dim"]
        ONLINE["Online Softmax<br/>m_new = max(m_prev, tile_max)<br/>l_new = rescale × l_prev + tile_sum"]
        ACCUM["O += P @ V<br/>(rescaled)"]
    end

    subgraph OUTPUT["Output"]
        NORM["Normalize<br/>O = O / l"]
        OUT["O[1, head_dim]"]
    end

    Q --> TILING
    K --> TILING
    V --> TILING

    TILE1 --> LOAD
    TILE2 --> LOAD
    TILE3 --> LOAD
    TILEN --> LOAD

    LOAD --> DOT --> SCALE --> ONLINE --> ACCUM
    ACCUM -->|"next tile"| LOAD
    ACCUM -->|"last tile"| NORM --> OUT
```

## Split-K Parallelism (Point 3)

```mermaid
flowchart TB
    subgraph STANDARD["Standard FA2 (1 block per head)"]
        direction LR
        S_HEAD["Head 0"]
        S_SEQ["Process seq[0:1024]<br/>sequentially"]
        S_OUT["Output"]
        S_HEAD --> S_SEQ --> S_OUT
    end

    subgraph SPLITK["Split-K FA2 (4 blocks per head)"]
        direction TB
        SK_HEAD["Head 0"]

        subgraph SPLITS["Parallel Splits"]
            direction LR
            SPLIT0["Block 0<br/>seq[0:256]"]
            SPLIT1["Block 1<br/>seq[256:512]"]
            SPLIT2["Block 2<br/>seq[512:768]"]
            SPLIT3["Block 3<br/>seq[768:1024]"]
        end

        subgraph PARTIAL["Partial Results"]
            direction LR
            P0["O_0, LSE_0"]
            P1["O_1, LSE_1"]
            P2["O_2, LSE_2"]
            P3["O_3, LSE_3"]
        end

        REDUCE_K["Reduce Kernel<br/>O = Σ(exp(LSE_i - max_LSE) × O_i)"]
        SK_OUT["Final Output"]

        SK_HEAD --> SPLITS
        SPLIT0 --> P0
        SPLIT1 --> P1
        SPLIT2 --> P2
        SPLIT3 --> P3
        P0 & P1 & P2 & P3 --> REDUCE_K --> SK_OUT
    end

    STANDARD -.->|"1 block = 1 head<br/>14 blocks for Qwen 0.5B"| SPLITK
    SPLITK -.->|"4 blocks per head<br/>56 blocks = better SM utilization"| SK_OUT
```

## Optimization Impact

```mermaid
graph LR
    subgraph BEFORE["Before Optimization"]
        B1["CUDA Cores<br/>20 TFLOPs"]
        B2["Per-token memcpy<br/>~0.6ms overhead"]
        B3["Sequential heads<br/>14 blocks"]
        B4["Sync softmax<br/>20% overhead"]
    end

    subgraph AFTER["After Optimization"]
        A1["Tensor Cores<br/>65 TFLOPs (T4)"]
        A2["GPU-Resident<br/>0ms overhead"]
        A3["Split-K<br/>56 blocks"]
        A4["Unified max<br/>async softmax"]
    end

    subgraph SPEEDUP["Expected Speedup"]
        S1["4-6x"]
        S2["1.5-2x"]
        S3["1.5-2x"]
        S4["1.15x"]
    end

    B1 -->|"Point 1"| A1 --> S1
    B2 -->|"Point 2"| A2 --> S2
    B3 -->|"Point 3"| A3 --> S3
    B4 -->|"Point 3"| A4 --> S4

    style BEFORE fill:#ffcdd2
    style AFTER fill:#c8e6c9
    style SPEEDUP fill:#bbdefb
```

## API Usage Flow

```mermaid
stateDiagram-v2
    [*] --> Uninitialized

    Uninitialized --> Initialized: flash_attention_v2_opt_init()
    note right of Initialized: GPU buffers allocated

    Initialized --> Ready: flash_attention_v2_opt_load_qkv()
    note right of Ready: Q, K, V on GPU

    Ready --> Computing: flash_attention_v2_opt_decode_gpu()
    Computing --> Ready: (auto-increments seq_len)

    Ready --> OutputReady: flash_attention_v2_opt_get_output()
    OutputReady --> Ready: (continue generation)

    Ready --> Initialized: flash_attention_v2_opt_reset()
    note right of Initialized: New generation

    Initialized --> Uninitialized: flash_attention_v2_opt_cleanup()
    OutputReady --> Uninitialized: flash_attention_v2_opt_cleanup()

    [*] --> Initialized: Backward-compatible API
    note left of Initialized: flash_attention_v2_opt_decode()<br/>handles init automatically
```

## File Structure

```
mojo-gateway/src/kernels/cuda/
├── flash_attention_v2.cu              # Original FA2 (baseline)
├── flash_attention_v2.h
├── flash_attention_v2_optimized.cu    # NEW: 3-point optimized
├── flash_attention_v2_optimized.h     # NEW: GPU-Resident API
├── test_flash_attention_v2_optimized.cu  # NEW: Frozen benchmark tests
└── Makefile                           # Updated with fa2-opt target

Build: make fa2-opt CUDA_ARCH="-gencode arch=compute_75,code=sm_75"
Test:  make test-fa2-opt
```
