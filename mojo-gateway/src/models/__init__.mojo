"""
Data Models for the Mojo Gateway.

This package provides type-safe request and response models
for all API endpoints.
"""

from .request import GenerateRequest, ChatRequest, ChatMessage, APIKeyRequest
from .response import (
    GenerateResponse,
    ChatResponse,
    ModelsResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse
)
