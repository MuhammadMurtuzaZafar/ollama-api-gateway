"""
HTTP Request Router for the Mojo Gateway.
Handles routing requests to appropriate handlers with middleware.
"""

from collections import Dict
from time import now

from lightbug_http import HTTPRequest, HTTPResponse, OK, NotFound
from lightbug_http.http import HTTPService

from .handlers.health import handle_health, handle_root, handle_readiness, handle_liveness
from .handlers.generate import handle_generate, GenerateHandlerResult
from .handlers.chat import handle_chat, handle_openai_chat, ChatHandlerResult
from .handlers.models import handle_list_models, handle_model_info, handle_openai_models, ModelsHandlerResult
from .handlers.keys import handle_create_key, handle_list_keys, handle_revoke_key, validate_api_key, KeysHandlerResult, KeyValidationResult
from .handlers.stats import handle_user_stats, handle_detailed_stats, handle_admin_stats, record_request_metrics, get_rate_limiter, StatsHandlerResult
from .models.response import ErrorResponse
from .middleware.rate_limiter import RateLimitResult
from .utils.config import get_config


@value
struct RouteMatch:
    """Result of route matching."""
    var matched: Bool
    var path_params: Dict[String, String]

    @staticmethod
    fn yes(params: Dict[String, String] = Dict[String, String]()) -> Self:
        return Self(matched=True, path_params=params)

    @staticmethod
    fn no() -> Self:
        return Self(matched=False, path_params=Dict[String, String]())


struct GatewayRouter(HTTPService):
    """
    Main HTTP request router for the Mojo Gateway.
    Implements the HTTPService trait for Lightbug HTTP server.
    """

    fn __init__(out self):
        pass

    fn func(mut self, req: HTTPRequest) raises -> HTTPResponse:
        """
        Route incoming HTTP requests to appropriate handlers.
        Implements authentication, rate limiting, and request logging.
        """
        var start_time = now()
        var path = String(req.uri.path)
        var method = String(req.method)

        # Log incoming request
        print("[" + method + "] " + path)

        # ===== Public endpoints (no auth required) =====
        if method == "GET":
            if path == "/":
                return self._json_response(handle_root(), 200)

            if path == "/health":
                return self._json_response(handle_health(), 200)

            if path == "/ready":
                return self._json_response(handle_readiness(), 200)

            if path == "/live":
                return self._json_response(handle_liveness(), 200)

        # ===== Protected endpoints (auth required) =====

        # Extract and validate API key
        var auth_header = self._get_header(req, "Authorization")
        var auth_result = validate_api_key(auth_header)

        if not auth_result.valid:
            return self._error_response(
                ErrorResponse(error=auth_result.error, code=401),
                401
            )

        var api_key = auth_result.api_key
        var is_admin = api_key.is_admin()

        # Check rate limit
        var rate_limiter = get_rate_limiter()
        var rate_result = rate_limiter.check_rate_limit(api_key.key, api_key.rate_limit)

        if not rate_result.allowed:
            var response = self._error_response(
                ErrorResponse(
                    error="Rate limit exceeded",
                    code=429,
                    details="Retry after " + String(rate_result.retry_after) + " seconds"
                ),
                429
            )
            # Add rate limit headers
            return self._add_rate_limit_headers(response, rate_result)

        # ===== Route to handlers =====
        var body = self._get_body(req)
        var response: HTTPResponse
        var tokens_generated = 0
        var model = String("")

        # Inference endpoints
        if path == "/api/generate" and method == "POST":
            var result = handle_generate(body)
            tokens_generated = result.tokens_generated
            response = self._json_response(result.response_body, result.status_code)

        elif path == "/api/chat" and method == "POST":
            var result = handle_chat(body)
            tokens_generated = result.tokens_generated
            response = self._json_response(result.response_body, result.status_code)

        elif path == "/v1/chat/completions" and method == "POST":
            var result = handle_openai_chat(body)
            tokens_generated = result.tokens_generated
            response = self._json_response(result.response_body, result.status_code)

        # Models endpoints
        elif path == "/api/models" and method == "GET":
            var result = handle_list_models()
            response = self._json_response(result.response_body, result.status_code)

        elif path == "/v1/models" and method == "GET":
            var result = handle_openai_models()
            response = self._json_response(result.response_body, result.status_code)

        elif path.startswith("/api/models/") and method == "GET":
            var model_name = path[12:]  # Remove "/api/models/"
            var result = handle_model_info(model_name)
            response = self._json_response(result.response_body, result.status_code)

        # API Key management endpoints
        elif path == "/api/keys" and method == "POST":
            var result = handle_create_key(body, is_admin)
            response = self._json_response(result.response_body, result.status_code)

        elif path == "/api/keys" and method == "GET":
            var result = handle_list_keys(is_admin)
            response = self._json_response(result.response_body, result.status_code)

        elif path.startswith("/api/keys/") and method == "DELETE":
            var key_preview = path[10:]  # Remove "/api/keys/"
            var result = handle_revoke_key(key_preview, is_admin)
            response = self._json_response(result.response_body, result.status_code)

        elif path.endswith("/reveal") and method == "GET":
            var key_name = path[10:len(path)-7]  # Extract key name
            var result = handle_revoke_key(key_name, is_admin)  # Reuse for now
            response = self._json_response(result.response_body, result.status_code)

        # Stats endpoints
        elif path == "/api/stats" and method == "GET":
            var result = handle_user_stats(api_key.get_preview())
            response = self._json_response(result.response_body, result.status_code)

        elif path == "/api/stats/detailed" and method == "GET":
            var result = handle_detailed_stats(api_key.get_preview())
            response = self._json_response(result.response_body, result.status_code)

        elif path == "/api/admin/stats" and method == "GET":
            var result = handle_admin_stats(is_admin)
            response = self._json_response(result.response_body, result.status_code)

        elif path == "/api/admin/stats/users" and method == "GET":
            var result = handle_admin_stats(is_admin)
            response = self._json_response(result.response_body, result.status_code)

        # 404 Not Found
        else:
            response = self._error_response(
                ErrorResponse(error="Not found: " + path, code=404),
                404
            )

        # Record metrics
        var end_time = now()
        var response_time_ms = Float64(end_time - start_time) / 1_000_000.0

        record_request_metrics(
            api_key=api_key.get_preview(),
            method=method,
            path=path,
            status_code=response.status_code,
            response_time_ms=response_time_ms,
            model=model,
            tokens_generated=tokens_generated
        )

        # Add rate limit headers
        response = self._add_rate_limit_headers(response, rate_result)

        return response

    fn _json_response(self, body: String, status_code: Int) -> HTTPResponse:
        """Create a JSON response."""
        var response = HTTPResponse(
            status_code=status_code,
            body=body.as_bytes()
        )
        response.headers.add("Content-Type", "application/json")
        response.headers.add("X-Powered-By", "Mojo-Gateway")
        return response

    fn _error_response(self, err: ErrorResponse, status_code: Int) -> HTTPResponse:
        """Create an error response."""
        return self._json_response(err.to_json(), status_code)

    fn _get_header(self, req: HTTPRequest, name: String) -> String:
        """Get a header value from the request."""
        # Lightbug HTTP header access
        try:
            return String(req.headers.get(name))
        except:
            return ""

    fn _get_body(self, req: HTTPRequest) -> String:
        """Get the request body as string."""
        if req.body_raw:
            return String(req.body_raw)
        return ""

    fn _add_rate_limit_headers(
        self,
        response: HTTPResponse,
        rate_result: RateLimitResult
    ) -> HTTPResponse:
        """Add rate limit headers to response."""
        var headers = rate_result.to_headers()
        for key in headers.keys():
            response.headers.add(key, headers[key])
        return response


# CORS middleware (simplified)
fn add_cors_headers(response: HTTPResponse) -> HTTPResponse:
    """Add CORS headers to allow cross-origin requests."""
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.headers.add("Access-Control-Max-Age", "86400")
    return response
