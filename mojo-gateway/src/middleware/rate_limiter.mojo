"""
High-performance rate limiter using sliding window algorithm.
Leverages Mojo's memory efficiency for tracking request counts.
"""

from collections import Dict, List
from time import now
from memory import UnsafePointer, memset_zero


@value
struct RateLimitEntry:
    """Single rate limit entry for a client."""
    var request_count: Int
    var window_start: Int    # Unix timestamp (seconds)
    var requests: List[Int]  # Timestamps of recent requests

    fn __init__(out self):
        self.request_count = 0
        self.window_start = Int(now() // 1_000_000_000)
        self.requests = List[Int]()


struct SlidingWindowRateLimiter:
    """
    Sliding window rate limiter implementation.
    More accurate than fixed window, less memory than sliding log.
    """
    var _entries: Dict[String, RateLimitEntry]
    var _window_seconds: Int
    var _cleanup_interval: Int
    var _last_cleanup: Int

    fn __init__(out self, window_seconds: Int = 3600):
        """
        Initialize rate limiter.

        Args:
            window_seconds: Size of the rate limiting window in seconds (default: 1 hour)
        """
        self._entries = Dict[String, RateLimitEntry]()
        self._window_seconds = window_seconds
        self._cleanup_interval = window_seconds // 4  # Cleanup every quarter window
        self._last_cleanup = Int(now() // 1_000_000_000)

    fn check_rate_limit(mut self, key: String, limit: Int) -> RateLimitResult:
        """
        Check if a request should be allowed under the rate limit.

        Args:
            key: Unique identifier for the client (API key)
            limit: Maximum requests allowed per window

        Returns:
            RateLimitResult indicating if request is allowed
        """
        var current_time = Int(now() // 1_000_000_000)

        # Periodic cleanup
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_entries(current_time)
            self._last_cleanup = current_time

        # Get or create entry for this key
        var entry: RateLimitEntry
        if key in self._entries:
            entry = self._entries[key]
        else:
            entry = RateLimitEntry()

        # Remove requests outside the window
        var window_start = current_time - self._window_seconds
        var valid_requests = List[Int]()
        for i in range(len(entry.requests)):
            if entry.requests[i] > window_start:
                valid_requests.append(entry.requests[i])

        entry.requests = valid_requests
        entry.request_count = len(valid_requests)

        # Check if under limit
        if entry.request_count >= limit:
            var retry_after = self._window_seconds - (current_time - entry.requests[0])
            self._entries[key] = entry
            return RateLimitResult.denied(
                limit=limit,
                remaining=0,
                retry_after=retry_after,
                reset_at=entry.requests[0] + self._window_seconds
            )

        # Add this request
        entry.requests.append(current_time)
        entry.request_count += 1
        self._entries[key] = entry

        return RateLimitResult.allowed(
            limit=limit,
            remaining=limit - entry.request_count,
            reset_at=current_time + self._window_seconds
        )

    fn get_usage(self, key: String) -> RateLimitUsage:
        """Get current rate limit usage for a key."""
        var current_time = Int(now() // 1_000_000_000)
        var window_start = current_time - self._window_seconds

        if key not in self._entries:
            return RateLimitUsage(requests_made=0, window_start=current_time)

        var entry = self._entries[key]
        var count = 0
        for i in range(len(entry.requests)):
            if entry.requests[i] > window_start:
                count += 1

        return RateLimitUsage(requests_made=count, window_start=window_start)

    fn reset(mut self, key: String):
        """Reset rate limit for a specific key."""
        if key in self._entries:
            _ = self._entries.pop(key)

    fn _cleanup_old_entries(mut self, current_time: Int):
        """Remove entries with no recent requests."""
        var window_start = current_time - self._window_seconds
        var keys_to_remove = List[String]()

        for key in self._entries.keys():
            var entry = self._entries[key]
            var has_valid = False
            for i in range(len(entry.requests)):
                if entry.requests[i] > window_start:
                    has_valid = True
                    break
            if not has_valid:
                keys_to_remove.append(key)

        for i in range(len(keys_to_remove)):
            _ = self._entries.pop(keys_to_remove[i])


@value
struct RateLimitResult:
    """Result of a rate limit check."""
    var allowed: Bool
    var limit: Int
    var remaining: Int
    var retry_after: Int    # Seconds until retry (if denied)
    var reset_at: Int       # Unix timestamp when limit resets

    @staticmethod
    fn allowed(limit: Int, remaining: Int, reset_at: Int) -> Self:
        return Self(
            allowed=True,
            limit=limit,
            remaining=remaining,
            retry_after=0,
            reset_at=reset_at
        )

    @staticmethod
    fn denied(limit: Int, remaining: Int, retry_after: Int, reset_at: Int) -> Self:
        return Self(
            allowed=False,
            limit=limit,
            remaining=remaining,
            retry_after=retry_after,
            reset_at=reset_at
        )

    fn to_headers(self) -> Dict[String, String]:
        """Generate rate limit headers for HTTP response."""
        var headers = Dict[String, String]()
        headers["X-RateLimit-Limit"] = String(self.limit)
        headers["X-RateLimit-Remaining"] = String(self.remaining)
        headers["X-RateLimit-Reset"] = String(self.reset_at)
        if not self.allowed:
            headers["Retry-After"] = String(self.retry_after)
        return headers


@value
struct RateLimitUsage:
    """Current rate limit usage statistics."""
    var requests_made: Int
    var window_start: Int

    fn to_json(self) -> String:
        return String(
            '{"requests_made":' + String(self.requests_made) + ','
            + '"window_start":' + String(self.window_start) + '}'
        )


# Token bucket rate limiter for burst handling
struct TokenBucketRateLimiter:
    """
    Token bucket rate limiter for handling burst traffic.
    Allows short bursts while maintaining average rate.
    """
    var _buckets: Dict[String, TokenBucket]
    var _capacity: Int
    var _refill_rate: Float64  # Tokens per second

    fn __init__(out self, capacity: Int = 100, refill_rate: Float64 = 10.0):
        """
        Initialize token bucket rate limiter.

        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self._buckets = Dict[String, TokenBucket]()
        self._capacity = capacity
        self._refill_rate = refill_rate

    fn try_consume(mut self, key: String, tokens: Int = 1) -> Bool:
        """
        Try to consume tokens from the bucket.

        Returns True if tokens were consumed, False if insufficient tokens.
        """
        var current_time = Float64(now()) / 1_000_000_000.0

        var bucket: TokenBucket
        if key in self._buckets:
            bucket = self._buckets[key]
        else:
            bucket = TokenBucket(
                tokens=Float64(self._capacity),
                last_refill=current_time,
                capacity=self._capacity
            )

        # Refill tokens based on elapsed time
        var elapsed = current_time - bucket.last_refill
        var refill_amount = elapsed * self._refill_rate
        bucket.tokens = min(Float64(bucket.capacity), bucket.tokens + refill_amount)
        bucket.last_refill = current_time

        # Check if we have enough tokens
        if bucket.tokens >= Float64(tokens):
            bucket.tokens -= Float64(tokens)
            self._buckets[key] = bucket
            return True

        self._buckets[key] = bucket
        return False

    fn get_tokens(self, key: String) -> Float64:
        """Get current token count for a key."""
        if key not in self._buckets:
            return Float64(self._capacity)
        return self._buckets[key].tokens


@value
struct TokenBucket:
    """Single token bucket for a client."""
    var tokens: Float64
    var last_refill: Float64  # Timestamp
    var capacity: Int


fn min(a: Float64, b: Float64) -> Float64:
    """Return minimum of two floats."""
    if a < b:
        return a
    return b
