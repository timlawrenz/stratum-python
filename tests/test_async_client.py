"""Tests for AsyncStratumClient."""

from __future__ import annotations

import httpx
import pytest
import respx

from stratum import AsyncStratumClient
from stratum.exceptions import JobFailedError, JobTimeoutError


@pytest.fixture
def async_client(api_key: str, base_url: str) -> AsyncStratumClient:
    return AsyncStratumClient(
        api_key=api_key,
        base_url=base_url,
        poll_interval=0.01,
        poll_max_interval=0.05,
        job_timeout=1.0,
        max_retries=1,
    )


class TestAsyncJobSubmission:
    @respx.mock
    @pytest.mark.asyncio
    async def test_submit_job(self, async_client, base_url, mock_job_response):
        respx.post(f"{base_url}/jobs").mock(
            return_value=httpx.Response(202, json=mock_job_response)
        )

        job = await async_client.jobs.submit(
            image_url="https://example.com/img.jpg",
            whole_image={"clip": True},
        )
        assert job.status == "queued"
        await async_client.close()


class TestAsyncJobPolling:
    @respx.mock
    @pytest.mark.asyncio
    async def test_wait_completed(self, async_client, base_url, job_id, mock_completed_job):
        respx.get(f"{base_url}/jobs/{job_id}").mock(
            return_value=httpx.Response(200, json=mock_completed_job)
        )

        job = await async_client.jobs.wait(job_id, timeout=5)
        assert job.status == "completed"
        await async_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_wait_timeout(self, async_client, base_url, job_id, mock_job_response):
        respx.get(f"{base_url}/jobs/{job_id}").mock(
            return_value=httpx.Response(200, json=mock_job_response)
        )

        with pytest.raises(JobTimeoutError):
            await async_client.jobs.wait(job_id, timeout=0.05)
        await async_client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_wait_failed(self, async_client, base_url, job_id, mock_job_response):
        failed = {**mock_job_response, "status": "failed", "error_message": "OOM"}
        respx.get(f"{base_url}/jobs/{job_id}").mock(
            return_value=httpx.Response(200, json=failed)
        )

        with pytest.raises(JobFailedError):
            await async_client.jobs.wait(job_id)
        await async_client.close()


class TestAsyncAnalyze:
    @respx.mock
    @pytest.mark.asyncio
    async def test_analyze(
        self, async_client, base_url, job_id, mock_completed_job, mock_clip_result
    ):
        respx.post(f"{base_url}/jobs").mock(
            return_value=httpx.Response(202, json=mock_completed_job)
        )
        respx.get(f"{base_url}/jobs/{job_id}").mock(
            return_value=httpx.Response(200, json=mock_completed_job)
        )
        respx.get(mock_completed_job["result_url"]).mock(
            return_value=httpx.Response(200, json=mock_clip_result)
        )

        results = await async_client.analyze(
            image_url="https://example.com/img.jpg",
            whole_image={"clip": True},
        )
        assert "whole_image.clip" in results



class TestAsyncContextManager:
    @respx.mock
    @pytest.mark.asyncio
    async def test_context_manager(self, api_key, base_url, mock_job_response):
        respx.post(f"{base_url}/jobs").mock(
            return_value=httpx.Response(202, json=mock_job_response)
        )

        async with AsyncStratumClient(
            api_key=api_key,
            base_url=base_url,
            max_retries=1,
        ) as client:
            job = await client.jobs.submit(
                image_url="https://example.com/img.jpg",
                whole_image={"clip": True},
            )
            assert job.status == "queued"
