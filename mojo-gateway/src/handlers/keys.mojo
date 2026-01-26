"""
API Key management endpoint handlers.
Handles /api/keys requests for key creation and management.
"""

from collections import List
from time import now

from ..models.request import APIKeyRequest
from ..models.response import ErrorResponse
from ..auth.api_key import APIKeyStore, APIKey, extract_api_key
from ..utils.json import parse_json, json_get_string, json_get_int


# Global API key store
var _key_store: APIKeyStore = APIKeyStore()


fn get_key_store() -> APIKeyStore:
    """Get the global API key store."""
    return _key_store


fn parse_api_key_request(body: String) raises -> APIKeyRequest:
    """Parse JSON body into APIKeyRequest."""
    var json = parse_json(body)

    var name = json_get_string(json, "name")
    var role = json_get_string(json, "role", "user")
    var rate_limit = json_get_int(json, "rate_limit", 100)

    return APIKeyRequest(name=name, role=role, rate_limit=rate_limit)


fn handle_create_key(body: String, is_admin: Bool) -> KeysHandlerResult:
    """
    Handle POST /api/keys endpoint.
    Creates a new API key (admin only).
    """
    if not is_admin:
        return KeysHandlerResult.error(
            ErrorResponse(error="Admin access required", code=403),
            403
        )

    # Parse request
    var request: APIKeyRequest
    try:
        request = parse_api_key_request(body)
    except e:
        return KeysHandlerResult.error(
            ErrorResponse(error="Invalid request: " + String(e), code=400),
            400
        )

    # Validate request
    try:
        _ = request.validate()
    except e:
        return KeysHandlerResult.error(
            ErrorResponse(error=String(e), code=400),
            400
        )

    # Create key
    try:
        var api_key = _key_store.create_key(
            name=request.name,
            role=request.role,
            rate_limit=request.rate_limit
        )

        # Return full key only on creation
        return KeysHandlerResult.success(api_key.to_json(include_full_key=True), 201)

    except e:
        return KeysHandlerResult.error(
            ErrorResponse(error=String(e), code=400),
            400
        )


fn handle_list_keys(is_admin: Bool) -> KeysHandlerResult:
    """
    Handle GET /api/keys endpoint.
    Lists all API keys (admin only).
    """
    if not is_admin:
        return KeysHandlerResult.error(
            ErrorResponse(error="Admin access required", code=403),
            403
        )

    var keys = _key_store.get_all_keys()

    var keys_json = String("[")
    for i in range(len(keys)):
        if i > 0:
            keys_json += ","
        keys_json += keys[i].to_json(include_full_key=False)
    keys_json += "]"

    return KeysHandlerResult.success('{"keys":' + keys_json + '}', 200)


fn handle_revoke_key(key_preview: String, is_admin: Bool) -> KeysHandlerResult:
    """
    Handle DELETE /api/keys/{key_preview} endpoint.
    Revokes an API key (admin only).
    """
    if not is_admin:
        return KeysHandlerResult.error(
            ErrorResponse(error="Admin access required", code=403),
            403
        )

    try:
        _ = _key_store.revoke_key(key_preview)
        return KeysHandlerResult.success('{"message":"API key revoked successfully"}', 200)
    except e:
        return KeysHandlerResult.error(
            ErrorResponse(error=String(e), code=404),
            404
        )


fn handle_reveal_key(key_id: String, is_admin: Bool) -> KeysHandlerResult:
    """
    Handle GET /api/keys/{id}/reveal endpoint.
    Reveals the full API key (admin only).
    """
    if not is_admin:
        return KeysHandlerResult.error(
            ErrorResponse(error="Admin access required", code=403),
            403
        )

    try:
        var api_key = _key_store.get_key_by_name(key_id)
        return KeysHandlerResult.success(api_key.to_json(include_full_key=True), 200)
    except e:
        return KeysHandlerResult.error(
            ErrorResponse(error=String(e), code=404),
            404
        )


fn validate_api_key(auth_header: String) -> KeyValidationResult:
    """
    Validate an API key from the Authorization header.
    Returns validation result with key metadata.
    """
    if len(auth_header) == 0:
        return KeyValidationResult.failure("Missing Authorization header")

    var key: String
    try:
        key = extract_api_key(auth_header)
    except e:
        return KeyValidationResult.failure("Invalid Authorization header format")

    try:
        var api_key = _key_store.validate_key(key)
        return KeyValidationResult.success(api_key)
    except e:
        return KeyValidationResult.failure(String(e))


@value
struct KeysHandlerResult:
    """Result of keys handler execution."""
    var success: Bool
    var response_body: String
    var status_code: Int

    @staticmethod
    fn success(body: String, code: Int) -> Self:
        return Self(success=True, response_body=body, status_code=code)

    @staticmethod
    fn error(err: ErrorResponse, code: Int) -> Self:
        return Self(success=False, response_body=err.to_json(), status_code=code)


@value
struct KeyValidationResult:
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
