"""Asynchronous Stratum API client."""

from __future__ import annotations

import asyncio
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
    _backoff_intervals,
    _handle_response,
    _should_retry,
)
from .exceptions import JobFailedError, JobTimeoutError
from .models import (
    AnalyzeImageRequest,
    BalanceResponse,
    FaceTasks,
    JobResponse,
    JobsPage,
    JobStatus,
    PersonTasks,
    SystemStatusResponse,
    WholeImageTasks,
)
from .results import JobResults, parse_job_results


def _convert_section(val: Any, model_class: type) -> Any:
    if isinstance(val, dict):
        return model_class(**val)
    return val


async def _async_request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    config: ClientConfig,
    **kwargs: Any,
) -> dict[str, Any]:
    """Make an async HTTP request with retry logic."""
    last_exc: Exception | None = None
    backoffs = _backoff_intervals()

    for attempt in range(config.max_retries + 1):
        try:
            response = await client.request(method, url, timeout=config.timeout, **kwargs)
            if _should_retry(response.status_code) and attempt < config.max_retries:
                wait = backoffs[min(attempt, len(backoffs) - 1)]
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            pass
                await asyncio.sleep(wait)
                continue
            return _handle_response(response)
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt < config.max_retries:
                await asyncio.sleep(backoffs[min(attempt, len(backoffs) - 1)])
                continue
            raise

    raise last_exc  # type: ignore[misc]


class _AsyncJobsNamespace:
    """Async job management methods."""

    def __init__(self, client: AsyncStratumClient) -> None:
        self._client = client

    async def submit(
        self,
        image_url: str,
        whole_image: WholeImageTasks | dict[str, bool] | None = None,
        prominent_person: PersonTasks | dict[str, bool] | None = None,
        prominent_face: FaceTasks | dict[str, bool] | None = None,
        callback_url: str | None = None,
    ) -> JobResponse:
        """Submit an image enrichment job."""
        body = AnalyzeImageRequest(
            image_url=image_url,
            whole_image=_convert_section(whole_image, WholeImageTasks),
            prominent_person=_convert_section(prominent_person, PersonTasks),
            prominent_face=_convert_section(prominent_face, FaceTasks),
            callback_url=callback_url,
        )
        data = await self._client._request("POST", "/jobs", json=body.model_dump())
        return JobResponse(**data)

    async def submit_file(
        self,
        file_path: str,
        whole_image: WholeImageTasks | dict[str, bool] | None = None,
        prominent_person: PersonTasks | dict[str, bool] | None = None,
        prominent_face: FaceTasks | dict[str, bool] | None = None,
        callback_url: str | None = None,
    ) -> JobResponse:
        """Submit an image file via multipart form upload asynchronously."""
        import json
        import os

        req_dict = AnalyzeImageRequest(
            image_url="placeholder",
            whole_image=_convert_section(whole_image, WholeImageTasks),
            prominent_person=_convert_section(prominent_person, PersonTasks),
            prominent_face=_convert_section(prominent_face, FaceTasks),
            callback_url=callback_url,
        ).model_dump(exclude={"image_url", "image_base64"}, exclude_none=True)

        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            files = {"image_file": (filename, f, "application/octet-stream")}
            data = {"request_json": json.dumps(req_dict)}
            resp_data = await self._client._request("POST", "/jobs/upload", files=files, data=data)

        return JobResponse(**resp_data)

    async def get(self, job_id: str | uuid.UUID) -> JobResponse:
        """Get current status of a job."""
        data = await self._client._request("GET", f"/jobs/{job_id}")
        return JobResponse(**data)

    async def wait(
        self,
        job_id: str | uuid.UUID,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> JobResponse:
        """Poll until a job completes or fails."""
        timeout = timeout or self._client._config.job_timeout
        interval = poll_interval or self._client._config.poll_interval
        max_interval = self._client._config.poll_max_interval
        elapsed = 0.0

        while True:
            job = await self.get(job_id)
            if job.status == "completed":
                return job
            if job.status == "failed":
                raise JobFailedError(str(job_id), job.error_message or "Unknown error")

            if elapsed >= timeout:
                raise JobTimeoutError(str(job_id), timeout)

            await asyncio.sleep(min(interval, max_interval))
            elapsed += interval
            interval = min(interval * 1.5, max_interval)

    async def results(
        self,
        job_id: str | uuid.UUID,
        _job: JobResponse | None = None,
    ) -> JobResults:
        """Download and parse results for a completed job."""
        job = _job or await self.get(job_id)
        if job.status != "completed":
            raise ValueError(f"Job {job_id} is not completed (status: {job.status})")
        if not job.result_url:
            raise ValueError(f"Job {job_id} has no result_url")

        for attempt in range(4):
            response = await self._client._http.get(
                job.result_url, timeout=self._client._config.timeout
            )
            if response.content:
                break
            if attempt < 3:
                await asyncio.sleep(1.5 ** attempt)
            else:
                raise RuntimeError(
                    f"Job {job_id} result_url returned empty body after {attempt + 1} attempts"
                )

        raw = response.json()
        return parse_job_results(raw)

    async def list(
        self, limit: int = 20, offset: int = 0, status: JobStatus | None = None
    ) -> JobsPage:
        """List jobs with optional status filter. Returns a paginated envelope."""
        params: dict = {"limit": limit, "offset": offset}
        if status is not None:
            params["status"] = status
        data = await self._client._request(
            "GET", "/jobs", params=params
        )
        return JobsPage(**data)


class _AsyncBillingNamespace:
    """Async billing methods."""

    def __init__(self, client: AsyncStratumClient) -> None:
        self._client = client

    async def balance(self) -> int:
        """Get current credit balance."""
        data = await self._client._request("GET", "/billing/balance")
        return BalanceResponse(**data).credit_balance

    async def checkout(self, pack_id: str = "starter") -> str:
        """Create a Stripe checkout session. Returns checkout URL."""
        data = await self._client._request(
            "POST", "/billing/checkout", params={"pack_id": pack_id}
        )
        return data["checkout_url"]


class _AsyncBatchNamespace:
    """Async batch submission helpers."""

    def __init__(self, client: AsyncStratumClient) -> None:
        self._client = client

    async def submit(
        self,
        image_urls: list[str],
        whole_image: WholeImageTasks | dict[str, bool] | None = None,
        prominent_person: PersonTasks | dict[str, bool] | None = None,
        prominent_face: FaceTasks | dict[str, bool] | None = None,
        callback_url: str | None = None,
        max_concurrent: int = 10,
    ) -> list[JobResponse]:
        """Submit multiple images with the same operations."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _submit_one(url: str) -> JobResponse:
            async with semaphore:
                return await self._client.jobs.submit(
                    image_url=url,
                    whole_image=whole_image,
                    prominent_person=prominent_person,
                    prominent_face=prominent_face,
                    callback_url=callback_url,
                )

        return await asyncio.gather(*[_submit_one(url) for url in image_urls])

    async def wait_all(
        self,
        jobs: list[JobResponse],
        timeout: float = 600,
        on_progress: Any = None,
    ) -> list[JobResults]:
        """Wait for all jobs and return parsed results."""

        async def _wait_one(i: int, job: JobResponse) -> tuple[int, JobResults]:
            completed = await self._client.jobs.wait(job.job_id, timeout=timeout)
            result = await self._client.jobs.results(completed.job_id)
            if on_progress:
                on_progress(i + 1, len(jobs))
            return i, result

        pairs = await asyncio.gather(
            *[_wait_one(i, job) for i, job in enumerate(jobs)]
        )
        ordered: list[JobResults | None] = [None] * len(jobs)
        for i, result in pairs:
            ordered[i] = result
        return ordered  # type: ignore[return-value]


class AsyncStratumClient:
    """Asynchronous client for the Stratum image enrichment API."""

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
        self._http = httpx.AsyncClient(headers=self._config.headers)
        self.jobs = _AsyncJobsNamespace(self)
        self.billing = _AsyncBillingNamespace(self)
        self.batch = _AsyncBatchNamespace(self)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Make an authenticated async request with retry logic."""
        return await _async_request_with_retry(
            self._http, method, self._config.url(path), self._config, **kwargs
        )

    async def analyze(
        self,
        image_url: str,
        whole_image: WholeImageTasks | dict[str, bool] | None = None,
        prominent_person: PersonTasks | dict[str, bool] | None = None,
        prominent_face: FaceTasks | dict[str, bool] | None = None,
        timeout: float | None = None,
    ) -> JobResults:
        """One-liner: submit, wait, return typed results."""
        job = await self.jobs.submit(
            image_url=image_url,
            whole_image=whole_image,
            prominent_person=prominent_person,
            prominent_face=prominent_face,
        )
        job = await self.jobs.wait(job.job_id, timeout=timeout)
        return await self.jobs.results(job.job_id, _job=job)

    async def status(self) -> SystemStatusResponse:
        """Get system status."""
        data = await self._request("GET", "/status")
        return SystemStatusResponse(**data)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncStratumClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
