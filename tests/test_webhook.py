"""Tests for webhook signature verification."""

from __future__ import annotations

import hashlib
import hmac

from stratum.webhook import verify_signature


class TestVerifySignature:
    def test_valid_signature(self):
        payload = '{"job_id": "123", "status": "completed"}'
        secret = "whsec_test_secret"
        sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        assert verify_signature(payload, sig, secret) is True

    def test_invalid_signature(self):
        payload = '{"job_id": "123"}'
        secret = "whsec_test_secret"
        assert verify_signature(payload, "invalid_hex", secret) is False

    def test_bytes_payload(self):
        payload = b'{"status": "completed"}'
        secret = "whsec_test"
        sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert verify_signature(payload, sig, secret) is True

    def test_tampered_payload(self):
        original = '{"job_id": "123"}'
        secret = "whsec_test"
        sig = hmac.new(secret.encode(), original.encode(), hashlib.sha256).hexdigest()
        assert verify_signature('{"job_id": "456"}', sig, secret) is False
