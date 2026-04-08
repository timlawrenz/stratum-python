"""Synchronous Stratum API client."""

from __future__ import annotations

import time
import uuid
from typing import Any

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
    TaskRequest,
)
from .operations import OPERATIONS, validate_operations
from .results import JobResults, parse_job_results


class _JobsNamespace:
    """Job management methods."""

    def __init__(self, client: StratumClient) -> None:
        self._client = client

    def submit(
        self,
        image_url: str,
        tasks: list[dict[str, Any]] | list[TaskRequest],
        sla_lane: str = "within_minutes",
        callback_url: str | None = None,
    ) -> JobResponse:
        """Submit an image enrichment job.

        Args:
            image_url: URL of the image to process.
            tasks: List of task dicts or TaskRequest objects.
            sla_lane: SLA lane (within_seconds, within_minutes, within_hours).
            callback_url: Optional webhook URL for completion notification.

        Returns:
            JobResponse with job_id and initial status.
        """
        task_objs = []
        for t in tasks:
            if isinstance(t, TaskRequest):
                task_objs.append(t)
            else:
                task_objs.append(TaskRequest(**t))

        body = AnalyzeImageRequest(
            image_url=image_url,
            tasks=task_objs,
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
        """Poll until a job completes or fails.

        Args:
            job_id: Job UUID to wait for.
            timeout: Max seconds to wait (default: client job_timeout).
            poll_interval: Initial poll interval in seconds.

        Returns:
            Completed JobResponse.

        Raises:
            JobFailedError: If the job fails.
            JobTimeoutError: If timeout is reached.
        """
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
        task_type_map: dict[str, str] | None = None,
    ) -> JobResults:
        """Download and parse results for a completed job.

        Args:
            job_id: Job UUID.
            task_type_map: Optional mapping of operation_id → operation type
                for result parsing. If not provided, operation_id is used.

        Returns:
            Parsed JobResults with typed result objects.
        """
        job = self.get(job_id)
        if job.status != "completed":
            raise ValueError(f"Job {job_id} is not completed (status: {job.status})")
        if not job.result_url:
            raise ValueError(f"Job {job_id} has no result_url")

        # Download from pre-signed R2 URL
        response = self._client._http.get(job.result_url, timeout=self._client._config.timeout)
        raw = response.json()
        return parse_job_results(raw, task_type_map)

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
        """Create a Stripe checkout session.

        Returns:
            Checkout URL to redirect user to.
        """
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
        return {k: v.model_dump() for k, v in resp.operations.items()}

    def list_local(self) -> dict[str, Any]:
        """List operations from the local SDK registry (no API call)."""
        return {k: v.model_dump() for k, v in OPERATIONS.items()}


class _BatchNamespace:
    """Batch submission helpers."""

    def __init__(self, client: StratumClient) -> None:
        self._client = client

    def submit(
        self,
        image_urls: list[str],
        operations: list[str],
        sla_lane: str = "within_minutes",
        callback_url: str | None = None,
    ) -> list[JobResponse]:
        """Submit multiple images with the same operations.

        Args:
            image_urls: List of image URLs to process.
            operations: Operation types to run on each image.
            sla_lane: SLA lane for all jobs.
            callback_url: Optional webhook URL.

        Returns:
            List of JobResponse objects.
        """
        validate_operations(operations)
        jobs = []
        for url in image_urls:
            tasks = [
                TaskRequest(
                    operation_id=op,
                    type=op,
                    params={"target": OPERATIONS[op].default_target},
                )
                for op in operations
            ]
            job = self._client.jobs.submit(
                image_url=url,
                tasks=tasks,
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
        """Wait for all jobs to complete and return parsed results.

        Args:
            jobs: List of submitted JobResponse objects.
            timeout: Total timeout for all jobs.
            on_progress: Optional callback(completed, total) called on each completion.

        Returns:
            List of parsed JobResults in same order as input.
        """
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
    """Synchronous client for the Stratum image enrichment API.

    Usage::

        client = StratumClient(api_key="sk_...")
        results = client.analyze(
            image_url="https://example.com/photo.jpg",
            operations=["embed_clip_vit_b_32"],
        )
        print(results["embed_clip_vit_b_32"].parsed.embedding)
    """

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
        """Make an authenticated request with retry logic."""
        return sync_request_with_retry(
            self._http, method, self._config.url(path), self._config, **kwargs
        )

    def analyze(
        self,
        image_url: str,
        operations: list[str],
        target: str | None = None,
        sla_lane: str = "within_minutes",
        timeout: float | None = None,
    ) -> JobResults:
        """One-liner: submit job, wait for completion, return typed results.

        Args:
            image_url: URL of the image to process.
            operations: List of operation types to run.
            target: Target for all operations (default: each op's default).
            sla_lane: SLA lane.
            timeout: Max seconds to wait for completion.

        Returns:
            Parsed JobResults with typed result objects.
        """
        validate_operations(operations)

        tasks = []
        type_map: dict[str, str] = {}
        for op in operations:
            op_info = OPERATIONS[op]
            tgt = target or op_info.default_target
            task = TaskRequest(
                operation_id=op,
                type=op,
                params={"target": tgt},
            )
            tasks.append(task)
            type_map[op] = op

        job = self.jobs.submit(image_url=image_url, tasks=tasks, sla_lane=sla_lane)
        job = self.jobs.wait(job.job_id, timeout=timeout)
        return self.jobs.results(job.job_id, task_type_map=type_map)

    def status(self) -> SystemStatusResponse:
        """Get system status (queue depths, SLA health)."""
        data = self._request("GET", "/status")
        return SystemStatusResponse(**data)

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    def __enter__(self) -> StratumClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
