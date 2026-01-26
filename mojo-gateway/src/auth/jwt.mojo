"""
JWT (JSON Web Token) authentication module.
Handles token generation, validation, and parsing.
"""

from collections import Dict
from base64 import b64encode, b64decode
from hashlib import sha256
from time import now

from ..utils.json import parse_json, json_get_string, json_get_int


@value
struct JWTClaims:
    """JWT token claims/payload."""
    var sub: String        # Subject (user/key identifier)
    var role: String       # User role (admin/user)
    var exp: Int           # Expiration timestamp
    var iat: Int           # Issued at timestamp
    var rate_limit: Int    # Rate limit for this token

    fn __init__(
        out self,
        sub: String,
        role: String = "user",
        exp: Int = 0,
        iat: Int = 0,
        rate_limit: Int = 100
    ):
        self.sub = sub
        self.role = role
        self.exp = exp if exp > 0 else now() + 86400  # Default: 24 hours
        self.iat = iat if iat > 0 else now()
        self.rate_limit = rate_limit

    fn is_expired(self) -> Bool:
        """Check if the token has expired."""
        return now() > self.exp

    fn is_admin(self) -> Bool:
        """Check if the token has admin privileges."""
        return self.role == "admin"

    fn to_json(self) -> String:
        """Serialize claims to JSON."""
        return String(
            '{"sub":"' + self.sub + '",'
            + '"role":"' + self.role + '",'
            + '"exp":' + String(self.exp) + ','
            + '"iat":' + String(self.iat) + ','
            + '"rate_limit":' + String(self.rate_limit) + '}'
        )


struct JWTValidator:
    """
    JWT token validator using HMAC-SHA256.
    """
    var _secret: String
    var _algorithm: String

    fn __init__(out self, secret: String):
        """Initialize with a secret key."""
        self._secret = secret
        self._algorithm = "HS256"

    fn create_token(self, claims: JWTClaims) -> String:
        """
        Create a new JWT token from claims.
        Format: header.payload.signature
        """
        # Header
        var header = '{"alg":"HS256","typ":"JWT"}'
        var header_b64 = self._base64url_encode(header)

        # Payload
        var payload = claims.to_json()
        var payload_b64 = self._base64url_encode(payload)

        # Signature
        var message = header_b64 + "." + payload_b64
        var signature = self._hmac_sha256(message, self._secret)
        var signature_b64 = self._base64url_encode(signature)

        return message + "." + signature_b64

    fn validate_token(self, token: String) raises -> JWTClaims:
        """
        Validate a JWT token and extract claims.
        Raises an error if validation fails.
        """
        # Split token into parts
        var parts = self._split_token(token)
        if len(parts) != 3:
            raise Error("Invalid token format: expected 3 parts")

        var header_b64 = parts[0]
        var payload_b64 = parts[1]
        var signature_b64 = parts[2]

        # Verify signature
        var message = header_b64 + "." + payload_b64
        var expected_signature = self._base64url_encode(
            self._hmac_sha256(message, self._secret)
        )

        if signature_b64 != expected_signature:
            raise Error("Invalid token signature")

        # Decode payload
        var payload = self._base64url_decode(payload_b64)
        var claims_dict = parse_json(payload)

        # Extract claims
        var claims = JWTClaims(
            sub=json_get_string(claims_dict, "sub"),
            role=json_get_string(claims_dict, "role", "user"),
            exp=json_get_int(claims_dict, "exp"),
            iat=json_get_int(claims_dict, "iat"),
            rate_limit=json_get_int(claims_dict, "rate_limit", 100)
        )

        # Check expiration
        if claims.is_expired():
            raise Error("Token has expired")

        return claims

    fn extract_from_header(self, auth_header: String) raises -> String:
        """Extract token from Authorization header (Bearer scheme)."""
        if not auth_header.startswith("Bearer "):
            raise Error("Invalid authorization header format")
        return auth_header[7:]  # Remove "Bearer " prefix

    fn _split_token(self, token: String) -> List[String]:
        """Split token by '.' delimiter."""
        var parts = List[String]()
        var current = String("")

        for i in range(len(token)):
            var c = token[i]
            if c == '.':
                parts.append(current)
                current = String("")
            else:
                current += c

        if len(current) > 0:
            parts.append(current)

        return parts

    fn _base64url_encode(self, data: String) -> String:
        """
        Base64url encode (URL-safe base64 without padding).
        """
        var encoded = b64encode(data.as_bytes())
        var result = String("")

        for i in range(len(encoded)):
            var c = encoded[i]
            if c == '+':
                result += '-'
            elif c == '/':
                result += '_'
            elif c == '=':
                pass  # Remove padding
            else:
                result += c

        return result

    fn _base64url_decode(self, data: String) raises -> String:
        """
        Base64url decode.
        """
        # Convert URL-safe characters back
        var normalized = String("")
        for i in range(len(data)):
            var c = data[i]
            if c == '-':
                normalized += '+'
            elif c == '_':
                normalized += '/'
            else:
                normalized += c

        # Add padding if needed
        var padding = (4 - len(normalized) % 4) % 4
        for _ in range(padding):
            normalized += '='

        var decoded_bytes = b64decode(normalized)
        return String(decoded_bytes)

    fn _hmac_sha256(self, message: String, key: String) -> String:
        """
        Compute HMAC-SHA256 signature.
        Simplified implementation - use proper crypto library in production.
        """
        # In production, use proper HMAC implementation
        # This is a simplified placeholder
        var combined = key + message
        var hash = sha256(combined.as_bytes())
        return String(hash.data())


@value
struct AuthResult:
    """Result of authentication attempt."""
    var authenticated: Bool
    var claims: JWTClaims
    var error: String

    @staticmethod
    fn success(claims: JWTClaims) -> Self:
        return Self(authenticated=True, claims=claims, error="")

    @staticmethod
    fn failure(error: String) -> Self:
        return Self(
            authenticated=False,
            claims=JWTClaims(sub="", role=""),
            error=error
        )
