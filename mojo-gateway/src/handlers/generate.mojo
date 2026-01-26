"""
Text generation endpoint handler.
Handles /api/generate requests for text completion.
"""

from collections import List
from time import now

from ..models.request import GenerateRequest
from ..models.response import GenerateResponse, ErrorResponse
from ..inference.max_engine import get_inference_engine
from ..utils.json import parse_json, json_get_string, json_get_int, json_get_float, json_get_bool


fn parse_generate_request(body: String) raises -> GenerateRequest:
    """Parse JSON body into GenerateRequest."""
    var json = parse_json(body)

    var model = json_get_string(json, "model")
    var prompt = json_get_string(json, "prompt")
    var stream = json_get_bool(json, "stream", False)
    var temperature = json_get_float(json, "temperature", 0.7)
    var max_tokens = json_get_int(json, "max_tokens", 2048)
    var top_p = json_get_float(json, "top_p", 0.9)
    var top_k = json_get_int(json, "top_k", 40)

    # Handle Ollama-style num_predict parameter
    if max_tokens == 2048:
        var num_predict = json_get_int(json, "num_predict", 0)
        if num_predict > 0:
            max_tokens = num_predict

    return GenerateRequest(
        model=model,
        prompt=prompt,
        stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k
    )


fn handle_generate(body: String) -> GenerateHandlerResult:
    """
    Handle POST /api/generate endpoint.
    Performs text generation using MAX Engine.
    """
    var start_time = now()

    # Parse request
    var request: GenerateRequest
    try:
        request = parse_generate_request(body)
    except e:
        return GenerateHandlerResult.error(
            ErrorResponse(error="Invalid request: " + String(e), code=400),
            400
        )

    # Validate request
    try:
        _ = request.validate()
    except e:
        return GenerateHandlerResult.error(
            ErrorResponse(error=String(e), code=400),
            400
        )

    # Get inference engine
    var engine = get_inference_engine()
    if not engine:
        return GenerateHandlerResult.error(
            ErrorResponse(error="Inference engine not initialized", code=503),
            503
        )

    if not engine[0].is_ready():
        return GenerateHandlerResult.error(
            ErrorResponse(error="Inference engine not ready", code=503),
            503
        )

    # Perform inference
    try:
        var result = engine[0].generate(request)

        var response = GenerateResponse(
            model=request.model,
            response=result.text,
            done=True,
            total_duration_ns=result.total_time_ns,
            prompt_eval_count=result.prompt_tokens,
            eval_count=result.tokens_generated
        )

        return GenerateHandlerResult.success(
            response.to_json(),
            result.tokens_generated,
            Float64(now() - start_time) / 1_000_000.0  # Convert to ms
        )

    except e:
        return GenerateHandlerResult.error(
            ErrorResponse(error="Inference failed: " + String(e), code=500),
            500
        )


@value
struct GenerateHandlerResult:
    """Result of generate handler execution."""
    var success: Bool
    var response_body: String
    var status_code: Int
    var tokens_generated: Int
    var response_time_ms: Float64

    @staticmethod
    fn success(body: String, tokens: Int, time_ms: Float64) -> Self:
        return Self(
            success=True,
            response_body=body,
            status_code=200,
            tokens_generated=tokens,
            response_time_ms=time_ms
        )

    @staticmethod
    fn error(err: ErrorResponse, code: Int) -> Self:
        return Self(
            success=False,
            response_body=err.to_json(),
            status_code=code,
            tokens_generated=0,
            response_time_ms=0.0
        )


fn handle_generate_stream(body: String) raises:
    """
    Handle streaming generation (Server-Sent Events).

    In production, this would:
    1. Parse request
    2. Set up SSE connection
    3. Stream tokens as they're generated
    4. Send final [DONE] message
    """
    # Streaming would be implemented with:
    # - SSE (Server-Sent Events) protocol
    # - Chunked transfer encoding
    # - Token-by-token generation callbacks
    raise Error("Streaming not yet implemented in PoC")
