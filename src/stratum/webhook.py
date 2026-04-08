"""Webhook signature verification for Stratum callbacks."""

from __future__ import annotations

import hashlib
import hmac


def verify_signature(payload: bytes | str, signature: str, secret: str) -> bool:
    """Verify a Stratum webhook HMAC-SHA256 signature.

    Args:
        payload: Raw request body bytes (or string).
        signature: Value of X-Stratum-Signature header.
        secret: Your webhook secret.

    Returns:
        True if signature is valid.
    """
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
