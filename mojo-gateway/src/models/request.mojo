"""
Request models for the Mojo Gateway.
Type-safe structs for incoming API requests.
"""

from collections import List, Optional


@value
struct ChatMessage:
    """A single message in a chat conversation."""
    var role: String      # "system", "user", "assistant"
    var content: String

    fn __init__(out self, role: String, content: String):
        self.role = role
        self.content = content

    fn to_dict(self) -> Dict[String, String]:
        """Convert to dictionary for JSON serialization."""
        var d = Dict[String, String]()
        d["role"] = self.role
        d["content"] = self.content
        return d


@value
struct GenerateRequest:
    """Request model for text generation endpoint."""
    var model: String
    var prompt: String
    var stream: Bool
    var temperature: Float64
    var max_tokens: Int
    var top_p: Float64
    var top_k: Int
    var stop_sequences: List[String]

    fn __init__(
        out self,
        model: String,
        prompt: String,
        stream: Bool = False,
        temperature: Float64 = 0.7,
        max_tokens: Int = 2048,
        top_p: Float64 = 0.9,
        top_k: Int = 40,
    ):
        self.model = model
        self.prompt = prompt
        self.stream = stream
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = List[String]()

    fn validate(self) raises -> Bool:
        """Validate request parameters."""
        if len(self.model) == 0:
            raise Error("Model name is required")
        if len(self.prompt) == 0:
            raise Error("Prompt is required")
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise Error("Temperature must be between 0.0 and 2.0")
        if self.max_tokens < 1 or self.max_tokens > 128000:
            raise Error("max_tokens must be between 1 and 128000")
        return True


@value
struct ChatRequest:
    """Request model for chat completion endpoint."""
    var model: String
    var messages: List[ChatMessage]
    var stream: Bool
    var temperature: Float64
    var max_tokens: Int
    var top_p: Float64
    var presence_penalty: Float64
    var frequency_penalty: Float64

    fn __init__(
        out self,
        model: String,
        messages: List[ChatMessage],
        stream: Bool = False,
        temperature: Float64 = 0.7,
        max_tokens: Int = 2048,
        top_p: Float64 = 0.9,
        presence_penalty: Float64 = 0.0,
        frequency_penalty: Float64 = 0.0,
    ):
        self.model = model
        self.messages = messages
        self.stream = stream
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    fn validate(self) raises -> Bool:
        """Validate request parameters."""
        if len(self.model) == 0:
            raise Error("Model name is required")
        if len(self.messages) == 0:
            raise Error("At least one message is required")
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise Error("Temperature must be between 0.0 and 2.0")
        return True

    fn get_prompt_text(self) -> String:
        """Convert messages to a single prompt string for inference."""
        var prompt = String("")
        for i in range(len(self.messages)):
            var msg = self.messages[i]
            if msg.role == "system":
                prompt += "<|system|>\n" + msg.content + "\n"
            elif msg.role == "user":
                prompt += "<|user|>\n" + msg.content + "\n"
            elif msg.role == "assistant":
                prompt += "<|assistant|>\n" + msg.content + "\n"
        prompt += "<|assistant|>\n"
        return prompt


@value
struct APIKeyRequest:
    """Request model for API key creation."""
    var name: String
    var role: String  # "admin" or "user"
    var rate_limit: Int

    fn __init__(out self, name: String, role: String = "user", rate_limit: Int = 100):
        self.name = name
        self.role = role
        self.rate_limit = rate_limit

    fn validate(self) raises -> Bool:
        """Validate request parameters."""
        if len(self.name) == 0:
            raise Error("API key name is required")
        if self.role != "admin" and self.role != "user":
            raise Error("Role must be 'admin' or 'user'")
        if self.rate_limit < 1:
            raise Error("Rate limit must be positive")
        return True
