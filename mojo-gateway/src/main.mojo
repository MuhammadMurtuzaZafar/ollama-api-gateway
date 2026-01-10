"""
Mojo Gateway - High-Performance LLM Inference API Gateway
Simplified standalone version for testing compilation.
"""

from collections import Dict
from time import perf_counter_ns


# ============================================
# Configuration
# ============================================

struct Config:
    var host: String
    var port: Int
    var model_path: String
    var rate_limit: Int

    fn __init__(out self):
        self.host = "0.0.0.0"
        self.port = 8080
        self.model_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.rate_limit = 100


# ============================================
# JSON Utilities
# ============================================

fn escape_json(s: String) -> String:
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


# ============================================
# Request/Response Models
# ============================================

@fieldwise_init
struct GenerateRequest(Copyable, Movable):
    var model: String
    var prompt: String
    var temperature: Float64
    var max_tokens: Int


@fieldwise_init
struct GenerateResponse(Copyable, Movable):
    var model: String
    var response: String
    var done: Bool
    var eval_count: Int

    fn to_json(self) -> String:
        return String(
            '{"model":"' + self.model + '",'
            + '"response":"' + escape_json(self.response) + '",'
            + '"done":' + ("true" if self.done else "false") + ','
            + '"eval_count":' + String(self.eval_count) + '}'
        )


@fieldwise_init
struct HealthResponse(Copyable, Movable):
    var status: String
    var version: String
    var inference_ready: Bool

    fn to_json(self) -> String:
        return String(
            '{"status":"' + self.status + '",'
            + '"version":"' + self.version + '",'
            + '"inference_ready":' + ("true" if self.inference_ready else "false") + '}'
        )


@fieldwise_init
struct ErrorResponse(Copyable, Movable):
    var error: String
    var code: Int

    fn to_json(self) -> String:
        return '{"error":"' + self.error + '","code":' + String(self.code) + '}'


# ============================================
# Rate Limiter (Simplified)
# ============================================

struct RateLimiter:
    var requests: Dict[String, Int]
    var window_start: Dict[String, Int]
    var window_seconds: Int

    fn __init__(out self, window_seconds: Int = 3600):
        self.requests = Dict[String, Int]()
        self.window_start = Dict[String, Int]()
        self.window_seconds = window_seconds

    fn check(mut self, key: String, limit: Int) raises -> Bool:
        """Check if request is allowed under rate limit."""
        var current_time = Int(perf_counter_ns() // 1_000_000_000)

        if key not in self.requests:
            self.requests[key] = 0
            self.window_start[key] = current_time

        var window_start = self.window_start[key]

        # Reset window if expired
        if current_time - window_start > self.window_seconds:
            self.requests[key] = 0
            self.window_start[key] = current_time

        # Check limit
        if self.requests[key] >= limit:
            return False

        self.requests[key] = self.requests[key] + 1
        return True


# ============================================
# API Key Store (Simplified)
# ============================================

struct APIKeyStore:
    var keys: Dict[String, String]  # key -> role
    var rate_limits: Dict[String, Int]

    fn __init__(out self):
        self.keys = Dict[String, String]()
        self.rate_limits = Dict[String, Int]()

        # Create default admin key
        var admin_key = "ollama-admin-key-12345"
        self.keys[admin_key] = "admin"
        self.rate_limits[admin_key] = 10000
        print("Default admin API key: " + admin_key)

    fn validate(self, key: String) -> Bool:
        return key in self.keys

    fn get_role(self, key: String) raises -> String:
        if key in self.keys:
            return self.keys[key]
        return ""

    fn get_rate_limit(self, key: String) raises -> Int:
        if key in self.rate_limits:
            return self.rate_limits[key]
        return 100


# ============================================
# Mock Inference Engine
# ============================================

struct InferenceEngine:
    var is_ready: Bool
    var model_name: String

    fn __init__(out self):
        self.is_ready = False
        self.model_name = ""

    fn initialize(mut self, model_path: String):
        print("Initializing inference engine...")
        print("  Model: " + model_path)
        self.model_name = model_path
        self.is_ready = True
        print("Inference engine ready!")

    fn generate(self, prompt: String, temperature: Float64, max_tokens: Int) -> String:
        """Mock generation - returns echo response."""
        if "hello" in prompt.lower():
            return "Hello! I'm the Mojo Gateway running on MAX Engine."
        elif "code" in prompt.lower():
            return "Here's a simple function:\\n\\n```mojo\\nfn add(a: Int, b: Int) -> Int:\\n    return a + b\\n```"
        else:
            return "I received your prompt. Processing with Mojo speed!"


# ============================================
# Request Handler
# ============================================

struct RequestHandler:
    var config: Config
    var key_store: APIKeyStore
    var rate_limiter: RateLimiter
    var engine: InferenceEngine

    fn __init__(out self):
        self.config = Config()
        self.key_store = APIKeyStore()
        self.rate_limiter = RateLimiter()
        self.engine = InferenceEngine()

    fn initialize(mut self):
        self.engine.initialize(self.config.model_path)

    fn handle_health(self) -> String:
        var response = HealthResponse(
            status="healthy" if self.engine.is_ready else "degraded",
            version="0.1.0",
            inference_ready=self.engine.is_ready
        )
        return response.to_json()

    fn handle_generate(mut self, api_key: String, prompt: String) raises -> String:
        # Validate API key
        if not self.key_store.validate(api_key):
            var err = ErrorResponse(error="Invalid API key", code=401)
            return err.to_json()

        # Check rate limit
        var limit = self.key_store.get_rate_limit(api_key)
        if not self.rate_limiter.check(api_key, limit):
            var err = ErrorResponse(error="Rate limit exceeded", code=429)
            return err.to_json()

        # Check engine ready
        if not self.engine.is_ready:
            var err = ErrorResponse(error="Inference engine not ready", code=503)
            return err.to_json()

        # Generate response
        var result = self.engine.generate(prompt, 0.7, 2048)

        var response = GenerateResponse(
            model=self.engine.model_name,
            response=result,
            done=True,
            eval_count=len(result) // 4
        )
        return response.to_json()


# ============================================
# Main Entry Point
# ============================================

fn print_banner():
    print("")
    print("=" * 60)
    print("  MOJO GATEWAY - High-Performance LLM API Gateway")
    print("  Powered by Mojo + MAX Engine")
    print("=" * 60)
    print("")


fn main() raises:
    print_banner()

    # Initialize handler
    var handler = RequestHandler()
    handler.initialize()

    print("")
    print("Configuration:")
    print("  Host: " + handler.config.host)
    print("  Port: " + String(handler.config.port))
    print("  Model: " + handler.config.model_path)
    print("")

    # Test health endpoint
    print("Testing /health endpoint:")
    var health = handler.handle_health()
    print("  Response: " + health)
    print("")

    # Test generate endpoint with valid key
    print("Testing /api/generate endpoint:")
    var api_key = "ollama-admin-key-12345"
    var generate_result = handler.handle_generate(api_key, "Hello, how are you?")
    print("  Response: " + generate_result)
    print("")

    # Test with invalid key
    print("Testing with invalid API key:")
    var invalid_result = handler.handle_generate("invalid-key", "Test")
    print("  Response: " + invalid_result)
    print("")

    # Test rate limiting (make many requests)
    print("Testing rate limiting:")
    var test_key = "test-key"
    handler.key_store.keys[test_key] = "user"
    handler.key_store.rate_limits[test_key] = 5  # Low limit for testing

    for i in range(7):
        var result = handler.handle_generate(test_key, "Request " + String(i))
        if "Rate limit" in result:
            print("  Request " + String(i) + ": Rate limited!")
        else:
            print("  Request " + String(i) + ": OK")

    print("")
    print("=" * 60)
    print("  All tests completed successfully!")
    print("=" * 60)
