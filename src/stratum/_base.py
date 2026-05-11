"""Shared HTTP logic for sync and async clients."""

from __future__ import annotations

import time
from typing import Any

import httpx

from .exceptions import raise_for_status

DEFAULT_BASE_URL = "https://stratum.pi216.ai"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_POLL_MAX_INTERVAL = 30.0
DEFAULT_JOB_TIMEOUT = 300.0


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _handle_response(response: httpx.Response) -> dict[str, Any]:
    """Parse response, raise on error."""
    if response.status_code >= 400:
        try:
            body = response.text
        except Exception:
            body = f"HTTP {response.status_code}"
        raise_for_status(
            response.status_code,
            body,
            headers=dict(response.headers),
        )
    if response.status_code == 204:
        return {}
    return response.json()


def _backoff_intervals() -> list[float]:
    """Generate exponential backoff intervals: 1, 2, 4, 8, ..."""
    return [min(2**i, 60) for i in range(10)]


class ClientConfig:
    """Shared configuration for Stratum clients."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        poll_max_interval: float = DEFAULT_POLL_MAX_INTERVAL,
        job_timeout: float = DEFAULT_JOB_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.poll_max_interval = poll_max_interval
        self.job_timeout = job_timeout

    @property
    def headers(self) -> dict[str, str]:
        return {
            **_auth_headers(self.api_key),
            "User-Agent": "stratum-python/0.1.0",
        }

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"


def _should_retry(status_code: int) -> bool:
    """Return True for retryable status codes."""
    return status_code in (429, 502, 503, 504)


def sync_request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    config: ClientConfig,
    **kwargs: Any,
) -> dict[str, Any]:
    """Make an HTTP request with retry logic."""
    last_exc: Exception | None = None
    backoffs = _backoff_intervals()

    for attempt in range(config.max_retries + 1):
        try:
            response = client.request(method, url, timeout=config.timeout, **kwargs)
            if _should_retry(response.status_code) and attempt < config.max_retries:
                wait = backoffs[min(attempt, len(backoffs) - 1)]
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            pass
                time.sleep(wait)
                continue
            return _handle_response(response)
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt < config.max_retries:
                time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
                continue
            raise

    raise last_exc  # type: ignore[misc]
