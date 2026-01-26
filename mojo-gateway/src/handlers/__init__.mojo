"""
HTTP Request Handlers for the Mojo Gateway.

This package contains all endpoint handlers:
- health: Health check and status endpoints
- generate: Text generation endpoint
- chat: Chat completion endpoint
- models: Model listing endpoint
- keys: API key management endpoints
- stats: Statistics and analytics endpoints
"""

from .health import handle_health, handle_root
from .generate import handle_generate
from .chat import handle_chat, handle_openai_chat
from .models import handle_list_models, handle_openai_models
from .keys import handle_create_key, handle_list_keys, validate_api_key
from .stats import handle_user_stats, handle_admin_stats
