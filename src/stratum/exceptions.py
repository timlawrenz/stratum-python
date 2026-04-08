"""Stratum API error hierarchy."""

from __future__ import annotations


class StratumError(Exception):
    """Base exception for all Stratum SDK errors."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(StratumError):
    """Invalid or missing API key (HTTP 401/403)."""


class InsufficientCreditsError(StratumError):
    """Not enough credits to process the request (HTTP 402)."""


class ValidationError(StratumError):
    """Invalid request parameters (HTTP 400/422)."""


class RateLimitError(StratumError):
    """Too many pending requests (HTTP 429).

    Attributes:
        retry_after: Seconds to wait before retrying, if provided by server.
    """

    def __init__(
        self, message: str, *, status_code: int = 429, retry_after: float | None = None
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.retry_after = retry_after


class JobFailedError(StratumError):
    """A job completed with status 'failed'.

    Attributes:
        job_id: The UUID of the failed job.
        error_message: Server-provided error description.
    """

    def __init__(self, job_id: str, error_message: str) -> None:
        super().__init__(f"Job {job_id} failed: {error_message}")
        self.job_id = job_id
        self.error_message = error_message


class JobTimeoutError(StratumError):
    """Timed out waiting for a job to complete.

    Attributes:
        job_id: The UUID of the timed-out job.
        timeout: Timeout value in seconds.
    """

    def __init__(self, job_id: str, timeout: float) -> None:
        super().__init__(f"Job {job_id} did not complete within {timeout}s")
        self.job_id = job_id
        self.timeout = timeout


class ServerError(StratumError):
    """Unexpected server error (HTTP 5xx)."""


_STATUS_MAP: dict[int, type[StratumError]] = {
    400: ValidationError,
    401: AuthenticationError,
    402: InsufficientCreditsError,
    403: AuthenticationError,
    422: ValidationError,
    429: RateLimitError,
}


def raise_for_status(status_code: int, body: str, headers: dict | None = None) -> None:
    """Raise the appropriate StratumError for an HTTP error response."""
    if 200 <= status_code < 300:
        return

    exc_cls = _STATUS_MAP.get(status_code)
    if exc_cls is RateLimitError:
        retry_after = None
        if headers and "retry-after" in headers:
            try:
                retry_after = float(headers["retry-after"])
            except (ValueError, TypeError):
                pass
        raise RateLimitError(body, retry_after=retry_after)

    if exc_cls:
        raise exc_cls(body, status_code=status_code)

    if status_code >= 500:
        raise ServerError(body, status_code=status_code)

    raise StratumError(body, status_code=status_code)
