"""
API Key management module.
Handles key generation, storage, and validation.
"""

from collections import Dict, List
from time import now
from random import random_si64

from ..utils.config import get_config


@value
struct APIKey:
    """Represents an API key with associated metadata."""
    var key: String
    var name: String
    var role: String          # "admin" or "user"
    var rate_limit: Int       # Requests per hour
    var is_active: Bool
    var created_at: Int       # Unix timestamp
    var last_used_at: Int     # Unix timestamp

    fn __init__(
        out self,
        key: String,
        name: String,
        role: String = "user",
        rate_limit: Int = 100,
        is_active: Bool = True
    ):
        self.key = key
        self.name = name
        self.role = role
        self.rate_limit = rate_limit
        self.is_active = is_active
        self.created_at = now()
        self.last_used_at = 0

    fn is_admin(self) -> Bool:
        """Check if this key has admin privileges."""
        return self.role == "admin"

    fn get_preview(self) -> String:
        """Get a preview of the key (first 12 characters)."""
        if len(self.key) > 12:
            return self.key[:12] + "..."
        return self.key

    fn to_json(self, include_full_key: Bool = False) -> String:
        """Serialize to JSON."""
        var key_value = self.get_preview() if not include_full_key else self.key
        return String(
            '{"key":"' + key_value + '",'
            + '"name":"' + self.name + '",'
            + '"role":"' + self.role + '",'
            + '"rate_limit":' + String(self.rate_limit) + ','
            + '"is_active":' + ("true" if self.is_active else "false") + ','
            + '"created_at":' + String(self.created_at) + ','
            + '"last_used_at":' + String(self.last_used_at) + '}'
        )


struct APIKeyStore:
    """
    In-memory API key store.
    In production, this would be backed by a database.
    """
    var _keys: Dict[String, APIKey]       # key -> APIKey
    var _names: Dict[String, String]      # name -> key (for uniqueness check)

    fn __init__(out self):
        self._keys = Dict[String, APIKey]()
        self._names = Dict[String, String]()

        # Create default admin key
        var admin_key = self._generate_key()
        var admin = APIKey(
            key=admin_key,
            name="default-admin",
            role="admin",
            rate_limit=10000,
            is_active=True
        )
        self._keys[admin_key] = admin
        self._names["default-admin"] = admin_key

        # Log the admin key (in production, handle securely)
        print("Default admin API key created: " + admin_key)

    fn create_key(
        mut self,
        name: String,
        role: String = "user",
        rate_limit: Int = 100
    ) raises -> APIKey:
        """Create a new API key."""
        # Check name uniqueness
        if name in self._names:
            raise Error("API key name already exists: " + name)

        # Validate role
        if role != "admin" and role != "user":
            raise Error("Role must be 'admin' or 'user'")

        # Generate new key
        var key = self._generate_key()
        var api_key = APIKey(
            key=key,
            name=name,
            role=role,
            rate_limit=rate_limit,
            is_active=True
        )

        self._keys[key] = api_key
        self._names[name] = key

        return api_key

    fn validate_key(mut self, key: String) raises -> APIKey:
        """Validate an API key and return its metadata."""
        if key not in self._keys:
            raise Error("Invalid API key")

        var api_key = self._keys[key]

        if not api_key.is_active:
            raise Error("API key is deactivated")

        # Update last used timestamp
        api_key.last_used_at = now()
        self._keys[key] = api_key

        return api_key

    fn revoke_key(mut self, key_preview: String) raises -> Bool:
        """Revoke (deactivate) an API key by its preview."""
        # Find key by preview
        for k in self._keys.keys():
            var api_key = self._keys[k]
            if api_key.get_preview() == key_preview or api_key.key == key_preview:
                api_key.is_active = False
                self._keys[k] = api_key
                return True

        raise Error("API key not found")

    fn get_all_keys(self) -> List[APIKey]:
        """Get all API keys (without full key values)."""
        var keys = List[APIKey]()
        for k in self._keys.keys():
            keys.append(self._keys[k])
        return keys

    fn get_key_by_name(self, name: String) raises -> APIKey:
        """Get an API key by its name."""
        if name not in self._names:
            raise Error("API key not found: " + name)

        var key = self._names[name]
        return self._keys[key]

    fn _generate_key(self) -> String:
        """Generate a cryptographically secure API key."""
        # Generate random bytes and encode as URL-safe string
        var chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        var key = "ollama-"

        for _ in range(32):
            var idx = Int(random_si64(0, len(chars) - 1))
            key += chars[idx]

        return key


@value
struct APIKeyValidationResult:
    """Result of API key validation."""
    var valid: Bool
    var api_key: APIKey
    var error: String

    @staticmethod
    fn success(api_key: APIKey) -> Self:
        return Self(valid=True, api_key=api_key, error="")

    @staticmethod
    fn failure(error: String) -> Self:
        return Self(
            valid=False,
            api_key=APIKey(key="", name="", role="user"),
            error=error
        )


fn extract_api_key(auth_header: String) raises -> String:
    """Extract API key from Authorization header."""
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    raise Error("Invalid authorization header format")
