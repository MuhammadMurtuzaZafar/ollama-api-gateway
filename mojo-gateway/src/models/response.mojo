"""
Response models for the Mojo Gateway.
Type-safe structs for API responses.
"""

from collections import List, Optional
from time import now


@value
struct GenerateResponse:
    """Response model for text generation endpoint."""
    var model: String
    var created_at: Int
    var response: String
    var done: Bool
    var total_duration_ns: Int
    var prompt_eval_count: Int
    var eval_count: Int

    fn __init__(
        out self,
        model: String,
        response: String,
        done: Bool = True,
        total_duration_ns: Int = 0,
        prompt_eval_count: Int = 0,
        eval_count: Int = 0,
    ):
        self.model = model
        self.created_at = now()
        self.response = response
        self.done = done
        self.total_duration_ns = total_duration_ns
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count

    fn to_json(self) -> String:
        """Serialize to JSON string."""
        return String(
            '{"model":"' + self.model + '",'
            + '"created_at":' + String(self.created_at) + ','
            + '"response":"' + self._escape_json(self.response) + '",'
            + '"done":' + ("true" if self.done else "false") + ','
            + '"total_duration":' + String(self.total_duration_ns) + ','
            + '"prompt_eval_count":' + String(self.prompt_eval_count) + ','
            + '"eval_count":' + String(self.eval_count) + '}'
        )

    @staticmethod
    fn _escape_json(s: String) -> String:
        """Escape special characters for JSON."""
        var result = String("")
        for i in range(len(s)):
            var c = s[i]
            if c == '"':
                result += '\\"'
            elif c == '\\':
                result += '\\\\'
            elif c == '\n':
                result += '\\n'
            elif c == '\r':
                result += '\\r'
            elif c == '\t':
                result += '\\t'
            else:
                result += c
        return result


@value
struct ChatChoice:
    """A single choice in chat completion response."""
    var index: Int
    var message: ChatMessageResponse
    var finish_reason: String

    fn to_json(self) -> String:
        return String(
            '{"index":' + String(self.index) + ','
            + '"message":' + self.message.to_json() + ','
            + '"finish_reason":"' + self.finish_reason + '"}'
        )


@value
struct ChatMessageResponse:
    """Message in chat completion response."""
    var role: String
    var content: String

    fn to_json(self) -> String:
        return String(
            '{"role":"' + self.role + '",'
            + '"content":"' + GenerateResponse._escape_json(self.content) + '"}'
        )


@value
struct ChatResponse:
    """Response model for chat completion endpoint (OpenAI-compatible)."""
    var id: String
    var object_type: String
    var created: Int
    var model: String
    var choices: List[ChatChoice]
    var usage: UsageStats

    fn __init__(
        out self,
        model: String,
        content: String,
        prompt_tokens: Int = 0,
        completion_tokens: Int = 0,
    ):
        self.id = "chatcmpl-" + String(now())
        self.object_type = "chat.completion"
        self.created = now()
        self.model = model
        self.choices = List[ChatChoice]()
        self.choices.append(ChatChoice(
            index=0,
            message=ChatMessageResponse(role="assistant", content=content),
            finish_reason="stop"
        ))
        self.usage = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )

    fn to_json(self) -> String:
        """Serialize to JSON string."""
        var choices_json = String("[")
        for i in range(len(self.choices)):
            if i > 0:
                choices_json += ","
            choices_json += self.choices[i].to_json()
        choices_json += "]"

        return String(
            '{"id":"' + self.id + '",'
            + '"object":"' + self.object_type + '",'
            + '"created":' + String(self.created) + ','
            + '"model":"' + self.model + '",'
            + '"choices":' + choices_json + ','
            + '"usage":' + self.usage.to_json() + '}'
        )


@value
struct UsageStats:
    """Token usage statistics."""
    var prompt_tokens: Int
    var completion_tokens: Int
    var total_tokens: Int

    fn to_json(self) -> String:
        return String(
            '{"prompt_tokens":' + String(self.prompt_tokens) + ','
            + '"completion_tokens":' + String(self.completion_tokens) + ','
            + '"total_tokens":' + String(self.total_tokens) + '}'
        )


@value
struct ModelInfo:
    """Information about an available model."""
    var name: String
    var size: Int
    var modified_at: String
    var digest: String

    fn to_json(self) -> String:
        return String(
            '{"name":"' + self.name + '",'
            + '"size":' + String(self.size) + ','
            + '"modified_at":"' + self.modified_at + '",'
            + '"digest":"' + self.digest + '"}'
        )


@value
struct ModelsResponse:
    """Response model for models list endpoint."""
    var models: List[ModelInfo]

    fn to_json(self) -> String:
        var models_json = String("[")
        for i in range(len(self.models)):
            if i > 0:
                models_json += ","
            models_json += self.models[i].to_json()
        models_json += "]"
        return '{"models":' + models_json + '}'


@value
struct HealthResponse:
    """Response model for health check endpoint."""
    var status: String
    var version: String
    var inference_ready: Bool
    var uptime_seconds: Int

    fn to_json(self) -> String:
        return String(
            '{"status":"' + self.status + '",'
            + '"version":"' + self.version + '",'
            + '"inference_ready":' + ("true" if self.inference_ready else "false") + ','
            + '"uptime_seconds":' + String(self.uptime_seconds) + '}'
        )


@value
struct ErrorResponse:
    """Response model for error responses."""
    var error: String
    var code: Int
    var details: String

    fn __init__(out self, error: String, code: Int = 400, details: String = ""):
        self.error = error
        self.code = code
        self.details = details

    fn to_json(self) -> String:
        var json = '{"error":"' + self.error + '","code":' + String(self.code)
        if len(self.details) > 0:
            json += ',"details":"' + self.details + '"'
        json += '}'
        return json
