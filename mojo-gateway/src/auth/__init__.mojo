"""
Authentication and Authorization module.

This package provides:
- JWT token generation and validation
- API key management
- Role-based access control
"""

from .jwt import JWTValidator, JWTClaims, AuthResult
from .api_key import APIKey, APIKeyStore, extract_api_key
