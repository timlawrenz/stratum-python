"""Tests for synchronous StratumClient."""

from __future__ import annotations

import httpx
import pytest
import respx

from stratum import StratumClient
from stratum.exceptions import (
    AuthenticationError,
    JobFailedError,
    JobTimeoutError,
    ValidationError,
)


@pytest.fixture
def client(api_key: str, base_url: str) -> StratumClient:
    return StratumClient(
        api_key=api_key,
        base_url=base_url,
        poll_interval=0.01,
        poll_max_interval=0.05,
        job_timeout=1.0,
        max_retries=1,
    )


class TestJobSubmission:
    @respx.mock
    def test_submit_job(self, client, base_url, mock_job_response):
        route = respx.post(f"{base_url}/jobs").mock(
            return_value=httpx.Response(202, json=mock_job_response)
        )

        job = client.jobs.submit(
            image_url="https://example.com/img.jpg",
            tasks=[{"type": "embed_clip_vit_b_32"}],
        )
        assert job.status == "queued"
        assert route.called

    @respx.mock
    def test_submit_with_task_objects(self, client, base_url, mock_job_response):
        from stratum.models import TaskRequest

        respx.post(f"{base_url}/jobs").mock(
            return_value=httpx.Response(202, json=mock_job_response)
        )

        job = client.jobs.submit(
            image_url="https://example.com/img.jpg",
            tasks=[TaskRequest(type="embed_clip_vit_b_32", params={"target": "whole_image"})],
        )
        assert job.status == "queued"


class TestJobPolling:
    @respx.mock
    def test_get_job(self, client, base_url, job_id, mock_job_response):
        respx.get(f"{base_url}/jobs/{job_id}").mock(
            return_value=httpx.Response(200, json=mock_job_response)
        )

        job = client.jobs.get(job_id)
        assert job.status == "queued"

    @respx.mock
    def test_wait_completed(self, client, base_url, job_id, mock_completed_job):
        respx.get(f"{base_url}/jobs/{job_id}").mock(
            return_value=httpx.Response(200, json=mock_completed_job)
        )

        job = client.jobs.wait(job_id, timeout=5)
        assert job.status == "completed"

    @respx.mock
    def test_wait_timeout(self, client, base_url, job_id, mock_job_response):
        respx.get(f"{base_url}/jobs/{job_id}").mock(
            return_value=httpx.Response(200, json=mock_job_response)
        )

        with pytest.raises(JobTimeoutError):
            client.jobs.wait(job_id, timeout=0.05)

    @respx.mock
    def test_wait_failed(self, client, base_url, job_id, mock_job_response):
        failed = {**mock_job_response, "status": "failed", "error_message": "CUDA OOM"}
        respx.get(f"{base_url}/jobs/{job_id}").mock(
            return_value=httpx.Response(200, json=failed)
        )

        with pytest.raises(JobFailedError) as exc_info:
            client.jobs.wait(job_id)
        assert "CUDA OOM" in str(exc_info.value)


class TestJobResults:
    @respx.mock
    def test_download_results(
        self, client, base_url, job_id, mock_completed_job, mock_clip_result
    ):
        respx.get(f"{base_url}/jobs/{job_id}").mock(
            return_value=httpx.Response(200, json=mock_completed_job)
        )
        respx.get(mock_completed_job["result_url"]).mock(
            return_value=httpx.Response(200, json=mock_clip_result)
        )

        results = client.jobs.results(job_id)
        assert "embed_clip_vit_b_32" in results
        assert results["embed_clip_vit_b_32"].parsed.dimensions == 512


class TestErrorHandling:
    @respx.mock
    def test_auth_error(self, client, base_url):
        respx.get(f"{base_url}/jobs").mock(
            return_value=httpx.Response(401, text="Invalid API key")
        )

        with pytest.raises(AuthenticationError):
            client.jobs.list()

    @respx.mock
    def test_rate_limit(self, client, base_url, mock_job_response):
        respx.post(f"{base_url}/jobs").mock(
            side_effect=[
                httpx.Response(429, text="Too many requests", headers={"retry-after": "0.01"}),
                httpx.Response(202, json=mock_job_response),
            ]
        )

        job = client.jobs.submit(
            image_url="https://example.com/img.jpg",
            tasks=[{"type": "embed_clip_vit_b_32"}],
        )
        assert job.status == "queued"

    @respx.mock
    def test_validation_error(self, client, base_url):
        respx.post(f"{base_url}/jobs").mock(
            return_value=httpx.Response(400, text="Invalid operation type")
        )

        with pytest.raises(ValidationError):
            client.jobs.submit(
                image_url="https://example.com/img.jpg",
                tasks=[{"type": "invalid_op"}],
            )


class TestAnalyze:
    @respx.mock
    def test_analyze_one_liner(
        self, client, base_url, job_id, mock_completed_job, mock_clip_result
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

        results = client.analyze(
            image_url="https://example.com/img.jpg",
            operations=["embed_clip_vit_b_32"],
        )
        assert "embed_clip_vit_b_32" in results

    def test_analyze_invalid_op(self, client):
        with pytest.raises(ValidationError):
            client.analyze(
                image_url="https://example.com/img.jpg",
                operations=["nonexistent_op"],
            )


class TestOperations:
    def test_list_local(self, client):
        ops = client.operations.list_local()
        assert "embed_clip_vit_b_32" in ops
        assert "embed_dino_v2" in ops
        assert ops["embed_clip_vit_b_32"]["credit_cost"] == 1


class TestBilling:
    @respx.mock
    def test_balance(self, client, base_url):
        respx.get(f"{base_url}/billing/balance").mock(
            return_value=httpx.Response(200, json={"credit_balance": 100})
        )

        balance = client.billing.balance()
        assert balance == 100
