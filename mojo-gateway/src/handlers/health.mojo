"""
Health check endpoint handler.
Provides system status and readiness information.
"""

from time import now

from ..models.response import HealthResponse, ErrorResponse
from ..inference.max_engine import get_inference_engine


# Server start time for uptime calculation
var _server_start_time: Int = 0


fn set_server_start_time():
    """Set the server start time (call at startup)."""
    global _server_start_time
    _server_start_time = now()


fn handle_health() -> String:
    """
    Handle GET /health endpoint.
    Returns server health status and inference engine readiness.
    """
    var engine = get_inference_engine()
    var inference_ready = False

    if engine:
        inference_ready = engine[0].is_ready()

    var uptime_seconds = 0
    if _server_start_time > 0:
        uptime_seconds = Int((now() - _server_start_time) // 1_000_000_000)

    var response = HealthResponse(
        status="healthy" if inference_ready else "degraded",
        version="0.1.0",
        inference_ready=inference_ready,
        uptime_seconds=uptime_seconds
    )

    return response.to_json()


fn handle_root() -> String:
    """
    Handle GET / endpoint.
    Returns basic service information.
    """
    return String(
        '{"service":"Ollama Mojo Gateway",'
        + '"version":"0.1.0",'
        + '"description":"High-performance API Gateway for LLM inference using Mojo and MAX Engine",'
        + '"endpoints":{'
        + '"/health":"Service health check",'
        + '"/api/generate":"Text generation",'
        + '"/api/chat":"Chat completions",'
        + '"/api/models":"List available models",'
        + '"/api/keys":"API key management",'
        + '"/api/stats":"Usage statistics"'
        + '}}'
    )


@value
struct ReadinessCheck:
    """Readiness check for Kubernetes/Docker health probes."""
    var database_connected: Bool
    var inference_ready: Bool
    var memory_ok: Bool

    fn is_ready(self) -> Bool:
        """Check if all systems are ready."""
        return self.inference_ready and self.memory_ok

    fn to_json(self) -> String:
        return String(
            '{"ready":' + ("true" if self.is_ready() else "false") + ','
            + '"checks":{'
            + '"inference":' + ("true" if self.inference_ready else "false") + ','
            + '"memory":' + ("true" if self.memory_ok else "false")
            + '}}'
        )


fn handle_readiness() -> String:
    """
    Handle GET /ready endpoint.
    Kubernetes readiness probe.
    """
    var engine = get_inference_engine()

    var check = ReadinessCheck(
        database_connected=True,  # Would check DB connection
        inference_ready=engine[0].is_ready() if engine else False,
        memory_ok=True  # Would check memory usage
    )

    return check.to_json()


fn handle_liveness() -> String:
    """
    Handle GET /live endpoint.
    Kubernetes liveness probe.
    """
    return '{"alive":true}'
