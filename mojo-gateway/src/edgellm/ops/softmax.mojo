"""
SIMD-optimized Softmax for EdgeLLM.

Numerically stable softmax:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Critical for attention mechanism - called once per head per layer.
"""

from memory import UnsafePointer
from algorithm import vectorize
from math import exp, inf


fn softmax(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    size: Int,
):
    """
    SIMD-vectorized numerically stable softmax.

    Args:
        output: Output buffer [size]
        input: Input buffer [size]
        size: Vector size
    """
    alias simd_width = simdwidthof[DType.float32]()

    # Phase 1: Find maximum for numerical stability
    var max_vec = SIMD[DType.float32, simd_width](-inf[DType.float32]())

    var i = 0
    while i + simd_width <= size:
        var v = input.load[width=simd_width](i)
        max_vec = max_vec.max(v)
        i += simd_width

    var max_val = max_vec.reduce_max()

    # Handle remainder
    while i < size:
        if input[i] > max_val:
            max_val = input[i]
        i += 1

    # Phase 2: Compute exp(x - max) and accumulate sum
    var sum_val: Float32 = 0.0

    i = 0
    while i + simd_width <= size:
        var v = input.load[width=simd_width](i)
        var shifted = v - max_val
        # Note: Mojo doesn't have vectorized exp yet, so we do scalar
        for j in range(simd_width):
            var exp_val = exp(shifted[j])
            output[i + j] = exp_val
            sum_val += exp_val
        i += simd_width

    # Handle remainder
    while i < size:
        var exp_val = exp(input[i] - max_val)
        output[i] = exp_val
        sum_val += exp_val
        i += 1

    # Phase 3: Normalize
    var inv_sum = 1.0 / sum_val
    var inv_sum_vec = SIMD[DType.float32, simd_width](inv_sum)

    i = 0
    while i + simd_width <= size:
        var v = output.load[width=simd_width](i)
        output.store[width=simd_width](i, v * inv_sum_vec)
        i += simd_width

    while i < size:
        output[i] *= inv_sum
        i += 1


fn softmax_inplace(
    x: UnsafePointer[Float32],
    size: Int,
):
    """
    In-place softmax (modifies input buffer).
    """
    # Find max
    var max_val: Float32 = x[0]
    for i in range(1, size):
        if x[i] > max_val:
            max_val = x[i]

    # Compute exp and sum
    var sum_val: Float32 = 0.0
    for i in range(size):
        x[i] = exp(x[i] - max_val)
        sum_val += x[i]

    # Normalize
    var inv_sum = 1.0 / sum_val
    for i in range(size):
        x[i] *= inv_sum


fn softmax_masked(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    mask: UnsafePointer[Bool],
    size: Int,
):
    """
    Masked softmax for causal attention.

    Positions where mask[i] = False are set to -inf before softmax.

    Args:
        output: Output buffer [size]
        input: Input buffer [size]
        mask: Boolean mask [size] - True = include, False = exclude
        size: Vector size
    """
    # Find max (only masked positions)
    var max_val: Float32 = -inf[DType.float32]()
    for i in range(size):
        if mask[i] and input[i] > max_val:
            max_val = input[i]

    # Handle case where all positions are masked
    if max_val == -inf[DType.float32]():
        for i in range(size):
            output[i] = 0.0
        return

    # Compute exp and sum (only masked positions)
    var sum_val: Float32 = 0.0
    for i in range(size):
        if mask[i]:
            output[i] = exp(input[i] - max_val)
            sum_val += output[i]
        else:
            output[i] = 0.0

    # Normalize
    if sum_val > 0:
        var inv_sum = 1.0 / sum_val
        for i in range(size):
            output[i] *= inv_sum


fn causal_softmax(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    size: Int,
    position: Int,
):
    """
    Causal softmax for autoregressive attention.

    Only considers positions <= current position.

    Args:
        output: Output buffer [size]
        input: Input buffer [size]
        size: Vector size (usually seq_len)
        position: Current position (0-indexed)
    """
    var valid_size = position + 1

    # Find max in valid range
    var max_val: Float32 = input[0]
    for i in range(1, valid_size):
        if input[i] > max_val:
            max_val = input[i]

    # Compute exp and sum
    var sum_val: Float32 = 0.0
    for i in range(valid_size):
        output[i] = exp(input[i] - max_val)
        sum_val += output[i]

    # Normalize valid positions
    var inv_sum = 1.0 / sum_val
    for i in range(valid_size):
        output[i] *= inv_sum

    # Zero out future positions
    for i in range(valid_size, size):
        output[i] = 0.0


fn log_softmax(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    size: Int,
):
    """
    Log-softmax: more numerically stable for cross-entropy loss.

    log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    """
    # Find max
    var max_val: Float32 = input[0]
    for i in range(1, size):
        if input[i] > max_val:
            max_val = input[i]

    # Compute sum of exp
    var sum_exp: Float32 = 0.0
    for i in range(size):
        sum_exp += exp(input[i] - max_val)

    var log_sum = log(sum_exp)

    # Compute log-softmax
    for i in range(size):
        output[i] = input[i] - max_val - log_sum


fn log(x: Float32) -> Float32:
    """Natural logarithm."""
    from math import log as math_log
    return math_log(x)


# ============================================================================
# Temperature-scaled softmax for sampling
# ============================================================================

fn softmax_temperature(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    size: Int,
    temperature: Float32 = 1.0,
):
    """
    Temperature-scaled softmax for controlling randomness in sampling.

    softmax(x/T) where T is temperature:
    - T < 1: More peaked distribution (more deterministic)
    - T = 1: Standard softmax
    - T > 1: Flatter distribution (more random)
    """
    if temperature <= 0:
        # Argmax behavior for T → 0
        var max_idx = 0
        var max_val = input[0]
        for i in range(1, size):
            if input[i] > max_val:
                max_val = input[i]
                max_idx = i

        for i in range(size):
            output[i] = 1.0 if i == max_idx else 0.0
        return

    var inv_temp = 1.0 / temperature

    # Find max
    var max_val: Float32 = input[0]
    for i in range(1, size):
        if input[i] > max_val:
            max_val = input[i]

    # Compute exp and sum
    var sum_val: Float32 = 0.0
    for i in range(size):
        output[i] = exp((input[i] - max_val) * inv_temp)
        sum_val += output[i]

    # Normalize
    var inv_sum = 1.0 / sum_val
    for i in range(size):
        output[i] *= inv_sum
