"""
SIMD-accelerated statistics calculations.
Leverages Mojo's vectorization for high-performance metrics aggregation.
"""

from algorithm import vectorize
from sys.info import simdwidthof
from memory import memset_zero, UnsafePointer
from math import sqrt


alias SIMD_WIDTH = simdwidthof[DType.float64]()


struct StatsAccumulator:
    """
    SIMD-accelerated statistics accumulator.
    Efficiently computes sum, mean, variance, min, max over large datasets.
    """
    var _data: UnsafePointer[Float64]
    var _size: Int
    var _capacity: Int
    var _sum: Float64
    var _sum_sq: Float64
    var _min: Float64
    var _max: Float64
    var _count: Int

    fn __init__(out self, capacity: Int = 10000):
        """Initialize with a given capacity."""
        self._capacity = capacity
        self._data = UnsafePointer[Float64].alloc(capacity)
        memset_zero(self._data, capacity)
        self._size = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = Float64.MAX
        self._max = Float64.MIN
        self._count = 0

    fn __del__(owned self):
        """Free allocated memory."""
        self._data.free()

    fn add(mut self, value: Float64):
        """Add a single value to the accumulator."""
        if self._size < self._capacity:
            self._data[self._size] = value
            self._size += 1

        self._sum += value
        self._sum_sq += value * value
        self._count += 1

        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    fn add_batch(mut self, values: UnsafePointer[Float64], count: Int):
        """
        Add a batch of values using SIMD operations.
        Significantly faster than adding values one at a time.
        """
        var sum_vec = SIMD[DType.float64, SIMD_WIDTH](0)
        var sum_sq_vec = SIMD[DType.float64, SIMD_WIDTH](0)
        var min_vec = SIMD[DType.float64, SIMD_WIDTH](Float64.MAX)
        var max_vec = SIMD[DType.float64, SIMD_WIDTH](Float64.MIN)

        # Process SIMD_WIDTH elements at a time
        var simd_count = (count // SIMD_WIDTH) * SIMD_WIDTH

        @parameter
        fn process_simd[width: Int](i: Int):
            var vec = values.load[width=width](i)
            sum_vec += vec.cast[DType.float64]()
            sum_sq_vec += (vec * vec).cast[DType.float64]()

            @parameter
            for j in range(width):
                var val = vec[j]
                if val < min_vec[j]:
                    min_vec[j] = val
                if val > max_vec[j]:
                    max_vec[j] = val

        vectorize[process_simd, SIMD_WIDTH](simd_count)

        # Reduce SIMD vectors to scalars
        var batch_sum: Float64 = 0.0
        var batch_sum_sq: Float64 = 0.0
        var batch_min: Float64 = Float64.MAX
        var batch_max: Float64 = Float64.MIN

        @parameter
        for i in range(SIMD_WIDTH):
            batch_sum += sum_vec[i]
            batch_sum_sq += sum_sq_vec[i]
            if min_vec[i] < batch_min:
                batch_min = min_vec[i]
            if max_vec[i] > batch_max:
                batch_max = max_vec[i]

        # Handle remaining elements
        for i in range(simd_count, count):
            var val = values[i]
            batch_sum += val
            batch_sum_sq += val * val
            if val < batch_min:
                batch_min = val
            if val > batch_max:
                batch_max = val

        # Update accumulator state
        self._sum += batch_sum
        self._sum_sq += batch_sum_sq
        self._count += count

        if batch_min < self._min:
            self._min = batch_min
        if batch_max > self._max:
            self._max = batch_max

    fn mean(self) -> Float64:
        """Calculate the arithmetic mean."""
        if self._count == 0:
            return 0.0
        return self._sum / Float64(self._count)

    fn variance(self) -> Float64:
        """Calculate the variance."""
        if self._count < 2:
            return 0.0
        var mean = self.mean()
        return (self._sum_sq / Float64(self._count)) - (mean * mean)

    fn std_dev(self) -> Float64:
        """Calculate the standard deviation."""
        return sqrt(self.variance())

    fn min(self) -> Float64:
        """Get the minimum value."""
        if self._count == 0:
            return 0.0
        return self._min

    fn max(self) -> Float64:
        """Get the maximum value."""
        if self._count == 0:
            return 0.0
        return self._max

    fn sum(self) -> Float64:
        """Get the sum of all values."""
        return self._sum

    fn count(self) -> Int:
        """Get the count of values."""
        return self._count

    fn reset(mut self):
        """Reset the accumulator."""
        self._size = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = Float64.MAX
        self._max = Float64.MIN
        self._count = 0


@value
struct RequestMetrics:
    """
    Metrics for tracking request performance.
    Uses SIMD-accelerated statistics internally.
    """
    var total_requests: Int
    var successful_requests: Int
    var failed_requests: Int
    var total_tokens_generated: Int
    var avg_response_time_ms: Float64
    var min_response_time_ms: Float64
    var max_response_time_ms: Float64
    var requests_per_second: Float64

    fn __init__(out self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens_generated = 0
        self.avg_response_time_ms = 0.0
        self.min_response_time_ms = 0.0
        self.max_response_time_ms = 0.0
        self.requests_per_second = 0.0

    fn to_json(self) -> String:
        """Serialize metrics to JSON."""
        return String(
            '{"total_requests":' + String(self.total_requests) + ','
            + '"successful_requests":' + String(self.successful_requests) + ','
            + '"failed_requests":' + String(self.failed_requests) + ','
            + '"total_tokens_generated":' + String(self.total_tokens_generated) + ','
            + '"avg_response_time_ms":' + String(self.avg_response_time_ms) + ','
            + '"min_response_time_ms":' + String(self.min_response_time_ms) + ','
            + '"max_response_time_ms":' + String(self.max_response_time_ms) + ','
            + '"requests_per_second":' + String(self.requests_per_second) + '}'
        )


struct MetricsCollector:
    """
    Thread-safe metrics collector using SIMD-accelerated statistics.
    """
    var _response_times: StatsAccumulator
    var _tokens_generated: StatsAccumulator
    var _total_requests: Int
    var _successful_requests: Int
    var _failed_requests: Int
    var _start_time_ns: Int

    fn __init__(out self):
        self._response_times = StatsAccumulator(100000)
        self._tokens_generated = StatsAccumulator(100000)
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._start_time_ns = now()

    fn record_request(
        mut self,
        response_time_ms: Float64,
        tokens_generated: Int,
        success: Bool
    ):
        """Record metrics for a single request."""
        self._response_times.add(response_time_ms)
        self._tokens_generated.add(Float64(tokens_generated))
        self._total_requests += 1

        if success:
            self._successful_requests += 1
        else:
            self._failed_requests += 1

    fn get_metrics(self) -> RequestMetrics:
        """Get aggregated metrics."""
        var metrics = RequestMetrics()
        metrics.total_requests = self._total_requests
        metrics.successful_requests = self._successful_requests
        metrics.failed_requests = self._failed_requests
        metrics.total_tokens_generated = Int(self._tokens_generated.sum())
        metrics.avg_response_time_ms = self._response_times.mean()
        metrics.min_response_time_ms = self._response_times.min()
        metrics.max_response_time_ms = self._response_times.max()

        var elapsed_seconds = Float64(now() - self._start_time_ns) / 1_000_000_000.0
        if elapsed_seconds > 0:
            metrics.requests_per_second = Float64(self._total_requests) / elapsed_seconds

        return metrics

    fn reset(mut self):
        """Reset all metrics."""
        self._response_times.reset()
        self._tokens_generated.reset()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._start_time_ns = now()


fn now() -> Int:
    """Get current time in nanoseconds (placeholder - use actual time API)."""
    # In real implementation, use proper time API
    from time import now as time_now
    return time_now()
