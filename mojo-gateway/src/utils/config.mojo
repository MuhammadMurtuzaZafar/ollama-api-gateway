"""
Configuration module for the Mojo Gateway.
Handles environment variables and runtime configuration.
"""

from collections import Dict
from sys import env_get_string


@value
struct Config:
    """Gateway configuration loaded from environment variables."""
    var host: String
    var port: Int
    var max_connections: Int
    var rate_limit_requests: Int
    var rate_limit_window_seconds: Int
    var jwt_secret: String
    var model_path: String
    var max_tokens_default: Int
    var temperature_default: Float64
    var log_level: String
    var enable_metrics: Bool

    fn __init__(out self):
        """Initialize configuration with defaults, overridden by environment."""
        self.host = env_get_string("GATEWAY_HOST", "0.0.0.0")
        self.port = 8080  # Default port
        self.max_connections = 1000
        self.rate_limit_requests = 100
        self.rate_limit_window_seconds = 3600
        self.jwt_secret = env_get_string("JWT_SECRET", "your-secret-key-change-in-production")
        self.model_path = env_get_string("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
        self.max_tokens_default = 2048
        self.temperature_default = 0.7
        self.log_level = env_get_string("LOG_LEVEL", "INFO")
        self.enable_metrics = True

    @staticmethod
    fn load() -> Config:
        """Load configuration from environment."""
        return Config()

    fn get_listen_address(self) -> String:
        """Get the full listen address string."""
        return self.host + ":" + String(self.port)


# Global configuration instance
var GLOBAL_CONFIG = Config()


fn get_config() -> Config:
    """Get the global configuration instance."""
    return GLOBAL_CONFIG
