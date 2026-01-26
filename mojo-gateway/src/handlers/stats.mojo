"""
Statistics endpoint handlers.
Handles /api/stats requests for usage analytics.
"""

from collections import Dict, List
from time import now

from ..models.response import ErrorResponse
from ..middleware.logging import RequestLogger, LogStats
from ..middleware.rate_limiter import SlidingWindowRateLimiter
from ..utils.simd_stats import MetricsCollector, RequestMetrics


# Global metrics collector
var _metrics_collector: MetricsCollector = MetricsCollector()

# Global request logger
var _request_logger: RequestLogger = RequestLogger()

# Global rate limiter
var _rate_limiter: SlidingWindowRateLimiter = SlidingWindowRateLimiter()


fn get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector


fn get_request_logger() -> RequestLogger:
    """Get the global request logger."""
    return _request_logger


fn get_rate_limiter() -> SlidingWindowRateLimiter:
    """Get the global rate limiter."""
    return _rate_limiter


fn handle_user_stats(api_key: String) -> StatsHandlerResult:
    """
    Handle GET /api/stats endpoint.
    Returns usage statistics for the authenticated user.
    """
    var usage = _rate_limiter.get_usage(api_key)
    var logs = _request_logger.get_logs_for_key(api_key, 100)

    # Calculate user-specific stats
    var total_requests = len(logs)
    var total_tokens = 0
    var total_time = 0.0

    for i in range(len(logs)):
        total_tokens += logs[i].tokens_generated
        total_time += logs[i].response_time_ms

    var avg_time = 0.0
    if total_requests > 0:
        avg_time = total_time / Float64(total_requests)

    var response = String(
        '{"api_key":"' + api_key[:12] + '...",'
        + '"total_requests":' + String(total_requests) + ','
        + '"total_tokens":' + String(total_tokens) + ','
        + '"avg_response_time_ms":' + String(avg_time) + ','
        + '"rate_limit_usage":' + usage.to_json() + '}'
    )

    return StatsHandlerResult.success(response)


fn handle_detailed_stats(api_key: String) -> StatsHandlerResult:
    """
    Handle GET /api/stats/detailed endpoint.
    Returns detailed usage statistics with breakdowns.
    """
    var logs = _request_logger.get_logs_for_key(api_key, 1000)

    # Group by endpoint
    var by_endpoint = Dict[String, Int]()
    # Group by model
    var by_model = Dict[String, Int]()

    var total_tokens = 0
    var total_time = 0.0
    var min_time = Float64.MAX
    var max_time = 0.0

    for i in range(len(logs)):
        var log = logs[i]

        # By endpoint
        if log.path in by_endpoint:
            by_endpoint[log.path] = by_endpoint[log.path] + 1
        else:
            by_endpoint[log.path] = 1

        # By model
        if len(log.model) > 0:
            if log.model in by_model:
                by_model[log.model] = by_model[log.model] + 1
            else:
                by_model[log.model] = 1

        total_tokens += log.tokens_generated
        total_time += log.response_time_ms

        if log.response_time_ms < min_time:
            min_time = log.response_time_ms
        if log.response_time_ms > max_time:
            max_time = log.response_time_ms

    var avg_time = 0.0
    if len(logs) > 0:
        avg_time = total_time / Float64(len(logs))

    # Build response
    var by_endpoint_json = String("{")
    var first = True
    for path in by_endpoint.keys():
        if not first:
            by_endpoint_json += ","
        by_endpoint_json += '"' + path + '":' + String(by_endpoint[path])
        first = False
    by_endpoint_json += "}"

    var by_model_json = String("{")
    first = True
    for model in by_model.keys():
        if not first:
            by_model_json += ","
        by_model_json += '"' + model + '":' + String(by_model[model])
        first = False
    by_model_json += "}"

    var response = String(
        '{"total_requests":' + String(len(logs)) + ','
        + '"total_tokens":' + String(total_tokens) + ','
        + '"response_time":{'
        + '"avg_ms":' + String(avg_time) + ','
        + '"min_ms":' + String(min_time if min_time != Float64.MAX else 0.0) + ','
        + '"max_ms":' + String(max_time)
        + '},'
        + '"by_endpoint":' + by_endpoint_json + ','
        + '"by_model":' + by_model_json + '}'
    )

    return StatsHandlerResult.success(response)


fn handle_admin_stats(is_admin: Bool) -> StatsHandlerResult:
    """
    Handle GET /api/admin/stats endpoint.
    Returns system-wide statistics (admin only).
    """
    if not is_admin:
        return StatsHandlerResult.error(
            ErrorResponse(error="Admin access required", code=403),
            403
        )

    var metrics = _metrics_collector.get_metrics()
    var log_stats = _request_logger.get_stats()

    var response = String(
        '{"system_metrics":' + metrics.to_json() + ','
        + '"log_stats":' + log_stats.to_json() + '}'
    )

    return StatsHandlerResult.success(response)


fn handle_admin_user_stats(is_admin: Bool) -> StatsHandlerResult:
    """
    Handle GET /api/admin/stats/users endpoint.
    Returns per-user statistics (admin only).
    """
    if not is_admin:
        return StatsHandlerResult.error(
            ErrorResponse(error="Admin access required", code=403),
            403
        )

    var logs = _request_logger.get_recent_logs(10000)

    # Group by API key
    var by_user = Dict[String, UserStats]()

    for i in range(len(logs)):
        var log = logs[i]
        var key = log.api_key

        if key in by_user:
            var stats = by_user[key]
            stats.total_requests += 1
            stats.total_tokens += log.tokens_generated
            stats.total_time += log.response_time_ms
            by_user[key] = stats
        else:
            var stats = UserStats(
                api_key=key,
                total_requests=1,
                total_tokens=log.tokens_generated,
                total_time=log.response_time_ms
            )
            by_user[key] = stats

    # Build response
    var users_json = String("[")
    var first = True
    for key in by_user.keys():
        if not first:
            users_json += ","
        users_json += by_user[key].to_json()
        first = False
    users_json += "]"

    return StatsHandlerResult.success('{"users":' + users_json + '}')


fn record_request_metrics(
    api_key: String,
    method: String,
    path: String,
    status_code: Int,
    response_time_ms: Float64,
    model: String = "",
    tokens_generated: Int = 0,
    error: String = ""
):
    """Record metrics for a request."""
    # Log the request
    _request_logger.log_request(
        api_key=api_key,
        method=method,
        path=path,
        status_code=status_code,
        response_time_ms=response_time_ms,
        model=model,
        tokens_generated=tokens_generated,
        error=error
    )

    # Update metrics collector
    _metrics_collector.record_request(
        response_time_ms=response_time_ms,
        tokens_generated=tokens_generated,
        success=(status_code >= 200 and status_code < 400)
    )


@value
struct UserStats:
    """Statistics for a single user."""
    var api_key: String
    var total_requests: Int
    var total_tokens: Int
    var total_time: Float64

    fn to_json(self) -> String:
        var avg_time = 0.0
        if self.total_requests > 0:
            avg_time = self.total_time / Float64(self.total_requests)

        return String(
            '{"api_key":"' + self.api_key + '",'
            + '"total_requests":' + String(self.total_requests) + ','
            + '"total_tokens":' + String(self.total_tokens) + ','
            + '"avg_response_time_ms":' + String(avg_time) + '}'
        )


@value
struct StatsHandlerResult:
    """Result of stats handler execution."""
    var success: Bool
    var response_body: String
    var status_code: Int

    @staticmethod
    fn success(body: String) -> Self:
        return Self(success=True, response_body=body, status_code=200)

    @staticmethod
    fn error(err: ErrorResponse, code: Int) -> Self:
        return Self(success=False, response_body=err.to_json(), status_code=code)
