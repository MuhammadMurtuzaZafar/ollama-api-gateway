"""
Request logging middleware.
Tracks all API requests for analytics and debugging.
"""

from collections import Dict, List
from time import now


@value
struct LogLevel:
    """Log level constants."""
    alias DEBUG = 0
    alias INFO = 1
    alias WARNING = 2
    alias ERROR = 3
    alias CRITICAL = 4

    var value: Int

    @staticmethod
    fn from_string(s: String) -> Self:
        if s == "DEBUG":
            return Self(value=Self.DEBUG)
        elif s == "INFO":
            return Self(value=Self.INFO)
        elif s == "WARNING":
            return Self(value=Self.WARNING)
        elif s == "ERROR":
            return Self(value=Self.ERROR)
        elif s == "CRITICAL":
            return Self(value=Self.CRITICAL)
        return Self(value=Self.INFO)

    fn to_string(self) -> String:
        if self.value == Self.DEBUG:
            return "DEBUG"
        elif self.value == Self.INFO:
            return "INFO"
        elif self.value == Self.WARNING:
            return "WARNING"
        elif self.value == Self.ERROR:
            return "ERROR"
        elif self.value == Self.CRITICAL:
            return "CRITICAL"
        return "UNKNOWN"


@value
struct RequestLog:
    """A single request log entry."""
    var timestamp: Int           # Unix timestamp (nanoseconds)
    var api_key: String          # API key used (preview only)
    var method: String           # HTTP method
    var path: String             # Request path
    var status_code: Int         # Response status code
    var response_time_ms: Float64  # Response time in milliseconds
    var model: String            # Model used (for inference endpoints)
    var tokens_generated: Int    # Tokens generated (for inference endpoints)
    var error: String            # Error message if any
    var client_ip: String        # Client IP address
    var user_agent: String       # User agent string

    fn __init__(
        out self,
        api_key: String,
        method: String,
        path: String,
        status_code: Int = 200,
        response_time_ms: Float64 = 0.0,
        model: String = "",
        tokens_generated: Int = 0,
        error: String = "",
        client_ip: String = "",
        user_agent: String = ""
    ):
        self.timestamp = now()
        self.api_key = api_key
        self.method = method
        self.path = path
        self.status_code = status_code
        self.response_time_ms = response_time_ms
        self.model = model
        self.tokens_generated = tokens_generated
        self.error = error
        self.client_ip = client_ip
        self.user_agent = user_agent

    fn to_json(self) -> String:
        """Serialize to JSON."""
        return String(
            '{"timestamp":' + String(self.timestamp) + ','
            + '"api_key":"' + self.api_key + '",'
            + '"method":"' + self.method + '",'
            + '"path":"' + self.path + '",'
            + '"status_code":' + String(self.status_code) + ','
            + '"response_time_ms":' + String(self.response_time_ms) + ','
            + '"model":"' + self.model + '",'
            + '"tokens_generated":' + String(self.tokens_generated) + ','
            + '"error":"' + self.error + '",'
            + '"client_ip":"' + self.client_ip + '",'
            + '"user_agent":"' + self.user_agent + '"}'
        )

    fn to_log_line(self) -> String:
        """Format as a log line for console output."""
        var timestamp_str = String(self.timestamp // 1_000_000)  # Convert to milliseconds
        var line = "[" + timestamp_str + "] "
        line += self.method + " " + self.path + " "
        line += String(self.status_code) + " "
        line += String(self.response_time_ms) + "ms"

        if len(self.model) > 0:
            line += " model=" + self.model

        if self.tokens_generated > 0:
            line += " tokens=" + String(self.tokens_generated)

        if len(self.error) > 0:
            line += " error=\"" + self.error + "\""

        return line


struct RequestLogger:
    """
    In-memory request logger with ring buffer for recent logs.
    """
    var _logs: List[RequestLog]
    var _max_logs: Int
    var _min_level: Int

    fn __init__(out self, max_logs: Int = 10000, min_level: String = "INFO"):
        """
        Initialize request logger.

        Args:
            max_logs: Maximum number of logs to keep in memory
            min_level: Minimum log level to record
        """
        self._logs = List[RequestLog]()
        self._max_logs = max_logs
        self._min_level = LogLevel.from_string(min_level).value

    fn log(mut self, entry: RequestLog):
        """Add a log entry."""
        # Rotate if at capacity
        if len(self._logs) >= self._max_logs:
            # Remove oldest 10%
            var remove_count = self._max_logs // 10
            var new_logs = List[RequestLog]()
            for i in range(remove_count, len(self._logs)):
                new_logs.append(self._logs[i])
            self._logs = new_logs

        self._logs.append(entry)

        # Also print to console
        print(entry.to_log_line())

    fn log_request(
        mut self,
        api_key: String,
        method: String,
        path: String,
        status_code: Int,
        response_time_ms: Float64,
        model: String = "",
        tokens_generated: Int = 0,
        error: String = "",
        client_ip: String = "",
        user_agent: String = ""
    ):
        """Log a request with individual parameters."""
        var entry = RequestLog(
            api_key=api_key,
            method=method,
            path=path,
            status_code=status_code,
            response_time_ms=response_time_ms,
            model=model,
            tokens_generated=tokens_generated,
            error=error,
            client_ip=client_ip,
            user_agent=user_agent
        )
        self.log(entry)

    fn get_recent_logs(self, count: Int = 100) -> List[RequestLog]:
        """Get the most recent log entries."""
        var result = List[RequestLog]()
        var start = max(0, len(self._logs) - count)

        for i in range(start, len(self._logs)):
            result.append(self._logs[i])

        return result

    fn get_logs_for_key(self, api_key: String, count: Int = 100) -> List[RequestLog]:
        """Get logs for a specific API key."""
        var result = List[RequestLog]()

        for i in range(len(self._logs) - 1, -1, -1):
            if self._logs[i].api_key == api_key:
                result.append(self._logs[i])
                if len(result) >= count:
                    break

        return result

    fn get_error_logs(self, count: Int = 100) -> List[RequestLog]:
        """Get recent error logs."""
        var result = List[RequestLog]()

        for i in range(len(self._logs) - 1, -1, -1):
            if self._logs[i].status_code >= 400:
                result.append(self._logs[i])
                if len(result) >= count:
                    break

        return result

    fn get_stats(self) -> LogStats:
        """Get aggregate statistics from logs."""
        var stats = LogStats()

        for i in range(len(self._logs)):
            var log = self._logs[i]
            stats.total_requests += 1
            stats.total_response_time += log.response_time_ms
            stats.total_tokens += log.tokens_generated

            if log.status_code >= 200 and log.status_code < 300:
                stats.successful_requests += 1
            elif log.status_code >= 400 and log.status_code < 500:
                stats.client_errors += 1
            elif log.status_code >= 500:
                stats.server_errors += 1

        if stats.total_requests > 0:
            stats.avg_response_time = stats.total_response_time / Float64(stats.total_requests)

        return stats

    fn clear(mut self):
        """Clear all logs."""
        self._logs = List[RequestLog]()


@value
struct LogStats:
    """Aggregate log statistics."""
    var total_requests: Int
    var successful_requests: Int
    var client_errors: Int
    var server_errors: Int
    var total_response_time: Float64
    var avg_response_time: Float64
    var total_tokens: Int

    fn __init__(out self):
        self.total_requests = 0
        self.successful_requests = 0
        self.client_errors = 0
        self.server_errors = 0
        self.total_response_time = 0.0
        self.avg_response_time = 0.0
        self.total_tokens = 0

    fn to_json(self) -> String:
        return String(
            '{"total_requests":' + String(self.total_requests) + ','
            + '"successful_requests":' + String(self.successful_requests) + ','
            + '"client_errors":' + String(self.client_errors) + ','
            + '"server_errors":' + String(self.server_errors) + ','
            + '"avg_response_time_ms":' + String(self.avg_response_time) + ','
            + '"total_tokens":' + String(self.total_tokens) + '}'
        )


fn max(a: Int, b: Int) -> Int:
    """Return maximum of two integers."""
    if a > b:
        return a
    return b
