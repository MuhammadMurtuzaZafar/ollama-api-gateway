"""
Chat completion endpoint handler.
Handles /api/chat requests for conversational AI.
"""

from collections import List
from time import now

from ..models.request import ChatRequest, ChatMessage
from ..models.response import ChatResponse, ErrorResponse
from ..inference.max_engine import get_inference_engine
from ..utils.json import parse_json, json_get_string, json_get_int, json_get_float, json_get_bool, JSONValue


fn parse_chat_request(body: String) raises -> ChatRequest:
    """Parse JSON body into ChatRequest."""
    var json = parse_json(body)

    var model = json_get_string(json, "model")
    var stream = json_get_bool(json, "stream", False)
    var temperature = json_get_float(json, "temperature", 0.7)
    var max_tokens = json_get_int(json, "max_tokens", 2048)
    var top_p = json_get_float(json, "top_p", 0.9)
    var presence_penalty = json_get_float(json, "presence_penalty", 0.0)
    var frequency_penalty = json_get_float(json, "frequency_penalty", 0.0)

    # Parse messages array
    var messages = parse_messages(body)

    return ChatRequest(
        model=model,
        messages=messages,
        stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty
    )


fn parse_messages(body: String) raises -> List[ChatMessage]:
    """
    Parse messages array from JSON body.
    Simple parser for the messages array structure.
    """
    var messages = List[ChatMessage]()

    # Find messages array
    var start_idx = body.find('"messages"')
    if start_idx == -1:
        raise Error("Missing 'messages' field")

    # Find array start
    var array_start = body.find('[', start_idx)
    if array_start == -1:
        raise Error("Invalid messages format")

    # Find array end
    var depth = 0
    var array_end = -1
    for i in range(array_start, len(body)):
        var c = body[i]
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                array_end = i
                break

    if array_end == -1:
        raise Error("Unterminated messages array")

    # Parse individual messages
    var messages_str = body[array_start+1:array_end]
    var current_pos = 0

    while current_pos < len(messages_str):
        # Find next object
        var obj_start = messages_str.find('{', current_pos)
        if obj_start == -1:
            break

        var obj_depth = 0
        var obj_end = -1
        for i in range(obj_start, len(messages_str)):
            var c = messages_str[i]
            if c == '{':
                obj_depth += 1
            elif c == '}':
                obj_depth -= 1
                if obj_depth == 0:
                    obj_end = i
                    break

        if obj_end == -1:
            break

        var msg_str = messages_str[obj_start:obj_end+1]
        var msg = parse_single_message(msg_str)
        messages.append(msg)

        current_pos = obj_end + 1

    return messages


fn parse_single_message(msg_str: String) raises -> ChatMessage:
    """Parse a single message object."""
    var json = parse_json(msg_str)
    var role = json_get_string(json, "role")
    var content = json_get_string(json, "content")

    if len(role) == 0:
        raise Error("Message missing 'role' field")
    if len(content) == 0:
        raise Error("Message missing 'content' field")

    return ChatMessage(role=role, content=content)


fn handle_chat(body: String) -> ChatHandlerResult:
    """
    Handle POST /api/chat endpoint.
    Performs chat completion using MAX Engine.
    """
    var start_time = now()

    # Parse request
    var request: ChatRequest
    try:
        request = parse_chat_request(body)
    except e:
        return ChatHandlerResult.error(
            ErrorResponse(error="Invalid request: " + String(e), code=400),
            400
        )

    # Validate request
    try:
        _ = request.validate()
    except e:
        return ChatHandlerResult.error(
            ErrorResponse(error=String(e), code=400),
            400
        )

    # Get inference engine
    var engine = get_inference_engine()
    if not engine:
        return ChatHandlerResult.error(
            ErrorResponse(error="Inference engine not initialized", code=503),
            503
        )

    if not engine[0].is_ready():
        return ChatHandlerResult.error(
            ErrorResponse(error="Inference engine not ready", code=503),
            503
        )

    # Perform inference
    try:
        var result = engine[0].chat(request)

        var response = ChatResponse(
            model=request.model,
            content=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.tokens_generated
        )

        return ChatHandlerResult.success(
            response.to_json(),
            result.tokens_generated,
            Float64(now() - start_time) / 1_000_000.0
        )

    except e:
        return ChatHandlerResult.error(
            ErrorResponse(error="Inference failed: " + String(e), code=500),
            500
        )


@value
struct ChatHandlerResult:
    """Result of chat handler execution."""
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


# OpenAI-compatible /v1/chat/completions endpoint
fn handle_openai_chat(body: String) -> ChatHandlerResult:
    """
    Handle POST /v1/chat/completions endpoint.
    OpenAI-compatible chat completions API.
    """
    # The format is the same as our chat endpoint, so we can reuse it
    return handle_chat(body)
