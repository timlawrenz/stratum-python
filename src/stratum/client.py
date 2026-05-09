"""Synchronous Stratum API client."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict

import httpx

from ._base import (
    DEFAULT_BASE_URL,
    DEFAULT_JOB_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_POLL_MAX_INTERVAL,
    DEFAULT_TIMEOUT,
    ClientConfig,
    sync_request_with_retry,
)
from .exceptions import JobFailedError, JobTimeoutError
from .models import (
    AnalyzeImageRequest,
    AvailableOperationsResponse,
    BalanceResponse,
    JobResponse,
    SystemStatusResponse,
    WholeImageTasks,
    PersonTasks,
    FaceTasks,
)
from .operations import list_operations, get_op_type
from .results import JobResults, parse_job_results


def _convert_section(val: Any, model_class: type) -> Any:
    if isinstance(val, dict):
        return model_class(**val)
    return val


class _JobsNamespace:
    """Job management methods."""

    def __init__(self, client: StratumClient) -> None:
        self._client = client

    def submit(
        self,
        image_url: str,
        whole_image: WholeImageTasks | dict[str, bool] | None = None,
        prominent_person: PersonTasks | dict[str, bool] | None = None,
        prominent_face: FaceTasks | dict[str, bool] | None = None,
        sla_lane: str = "within_minutes",
        callback_url: str | None = None,
    ) -> JobResponse:
        """Submit an image enrichment job.

        Args:
            image_url: URL of the image to process.
            whole_image: Operations to run on the full image.
            prominent_person: Operations to run on the prominent person.
            prominent_face: Operations to run on the prominent face.
            sla_lane: SLA lane (within_seconds, within_minutes, within_hours).
            callback_url: Optional webhook URL for completion notification.

        Returns:
            JobResponse with job_id and initial status.
        """
        body = AnalyzeImageRequest(
            image_url=image_url,
            whole_image=_convert_section(whole_image, WholeImageTasks),
            prominent_person=_convert_section(prominent_person, PersonTasks),
            prominent_face=_convert_section(prominent_face, FaceTasks),
            sla_lane=sla_lane,
            callback_url=callback_url,
        )
        data = self._client._request("POST", "/jobs", json=body.model_dump())
        return JobResponse(**data)

    def get(self, job_id: str | uuid.UUID) -> JobResponse:
        """Get current status of a job."""
        data = self._client._request("GET", f"/jobs/{job_id}")
        return JobResponse(**data)

    def wait(
        self,
        job_id: str | uuid.UUID,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> JobResponse:
        """Poll until a job completes or fails."""
        timeout = timeout or self._client._config.job_timeout
        interval = poll_interval or self._client._config.poll_interval
        max_interval = self._client._config.poll_max_interval
        start = time.monotonic()

        while True:
            job = self.get(job_id)
            if job.status == "completed":
                return job
            if job.status == "failed":
                raise JobFailedError(str(job_id), job.error_message or "Unknown error")

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise JobTimeoutError(str(job_id), timeout)

            time.sleep(min(interval, max_interval))
            interval = min(interval * 1.5, max_interval)

    def results(
        self,
        job_id: str | uuid.UUID,
    ) -> JobResults:
        """Download and parse results for a completed job."""
        job = self.get(job_id)
        if job.status != "completed":
            raise ValueError(f"Job {job_id} is not completed (status: {job.status})")
        if not job.result_url:
            raise ValueError(f"Job {job_id} has no result_url")

        response = self._client._http.get(job.result_url, timeout=self._client._config.timeout)
        raw = response.json()
        return parse_job_results(raw)

    def list(self, limit: int = 20, offset: int = 0) -> list[JobResponse]:
        """List recent jobs."""
        data = self._client._request(
            "GET", "/jobs", params={"limit": limit, "offset": offset}
        )
        return [JobResponse(**j) for j in data]


class _BillingNamespace:
    """Billing methods."""

    def __init__(self, client: StratumClient) -> None:
        self._client = client

    def balance(self) -> int:
        """Get current credit balance."""
        data = self._client._request("GET", "/billing/balance")
        return BalanceResponse(**data).credit_balance

    def checkout(self, pack_id: str = "starter") -> str:
        """Create a Stripe checkout session."""
        data = self._client._request(
            "POST", "/billing/checkout", params={"pack_id": pack_id}
        )
        return data["checkout_url"]


class _OperationsNamespace:
    """Operations discovery."""

    def __init__(self, client: StratumClient) -> None:
        self._client = client

    def list(self) -> dict[str, Any]:
        """List available operations from the server."""
        data = self._client._request("GET", "/available_operations")
        resp = AvailableOperationsResponse(**data)
        return resp.model_dump()

    def list_local(self) -> dict[str, list[str]]:
        """List operations from the local SDK registry."""
        return list_operations()


class _BatchNamespace:
    """Batch submission helpers."""

    def __init__(self, client: StratumClient) -> None:
        self._client = client

    def submit(
        self,
        image_urls: list[str],
        whole_image: WholeImageTasks | dict[str, bool] | None = None,
        prominent_person: PersonTasks | dict[str, bool] | None = None,
        prominent_face: FaceTasks | dict[str, bool] | None = None,
        sla_lane: str = "within_minutes",
        callback_url: str | None = None,
    ) -> list[JobResponse]:
        """Submit multiple images with the same operations."""
        jobs = []
        for url in image_urls:
            job = self._client.jobs.submit(
                image_url=url,
                whole_image=whole_image,
                prominent_person=prominent_person,
                prominent_face=prominent_face,
                sla_lane=sla_lane,
                callback_url=callback_url,
            )
            jobs.append(job)
        return jobs

    def wait_all(
        self,
        jobs: list[JobResponse],
        timeout: float = 600,
        on_progress: Any = None,
    ) -> list[JobResults]:
        """Wait for all jobs to complete and return parsed results."""
        start = time.monotonic()
        results: list[JobResults | None] = [None] * len(jobs)
        pending = set(range(len(jobs)))

        while pending:
            elapsed = time.monotonic() - start
            remaining = timeout - elapsed
            if remaining <= 0:
                pending_ids = [str(jobs[i].job_id) for i in pending]
                raise JobTimeoutError(
                    f"batch({len(pending_ids)} remaining)", timeout
                )

            for i in list(pending):
                job = self._client.jobs.get(jobs[i].job_id)
                if job.status == "completed":
                    results[i] = self._client.jobs.results(job.job_id)
                    pending.discard(i)
                    if on_progress:
                        on_progress(len(jobs) - len(pending), len(jobs))
                elif job.status == "failed":
                    raise JobFailedError(
                        str(job.job_id), job.error_message or "Unknown error"
                    )

            if pending:
                time.sleep(self._client._config.poll_interval)

        return results  # type: ignore[return-value]


class StratumClient:
    """Synchronous client for the Stratum image enrichment API."""

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
        self._config = ClientConfig(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            poll_interval=poll_interval,
            poll_max_interval=poll_max_interval,
            job_timeout=job_timeout,
        )
        self._http = httpx.Client(headers=self._config.headers)
        self.jobs = _JobsNamespace(self)
        self.billing = _BillingNamespace(self)
        self.operations = _OperationsNamespace(self)
        self.batch = _BatchNamespace(self)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return sync_request_with_retry(
            self._http, method, self._config.url(path), self._config, **kwargs
        )

    def analyze(
        self,
        image_url: str,
        whole_image: WholeImageTasks | dict[str, bool] | None = None,
        prominent_person: PersonTasks | dict[str, bool] | None = None,
        prominent_face: FaceTasks | dict[str, bool] | None = None,
        sla_lane: str = "within_minutes",
        timeout: float | None = None,
    ) -> JobResults:
        """One-liner: submit job, wait for completion, return typed results."""
        job = self.jobs.submit(
            image_url=image_url,
            whole_image=whole_image,
            prominent_person=prominent_person,
            prominent_face=prominent_face,
            sla_lane=sla_lane
        )
        job = self.jobs.wait(job.job_id, timeout=timeout)
        return self.jobs.results(job.job_id)

    def status(self) -> SystemStatusResponse:
        """Get system status (queue depths, SLA health)."""
        data = self._request("GET", "/status")
        return SystemStatusResponse(**data)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> StratumClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
