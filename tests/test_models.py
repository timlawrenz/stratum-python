"""Tests for Pydantic models."""

from __future__ import annotations

from stratum.models import (
    AnalyzeImageRequest,
    JobResponse,
    OperationInfo,
    TaskRequest,
)


class TestTaskRequest:
    def test_minimal(self):
        t = TaskRequest(type="embed_clip_vit_b_32")
        assert t.type == "embed_clip_vit_b_32"
        assert t.params == {}
        assert t.operation_id == ""

    def test_full(self):
        t = TaskRequest(
            operation_id="my_clip",
            type="embed_clip_vit_b_32",
            params={"target": "whole_image"},
        )
        assert t.operation_id == "my_clip"
        assert t.params["target"] == "whole_image"


class TestAnalyzeImageRequest:
    def test_defaults(self):
        req = AnalyzeImageRequest(
            image_url="https://example.com/img.jpg",
            tasks=[TaskRequest(type="embed_clip_vit_b_32")],
        )
        assert req.sla_lane == "within_minutes"
        assert req.callback_url is None

    def test_with_callback(self):
        req = AnalyzeImageRequest(
            image_url="https://example.com/img.jpg",
            tasks=[TaskRequest(type="embed_clip_vit_b_32")],
            callback_url="https://webhook.site/test",
        )
        assert req.callback_url == "https://webhook.site/test"


class TestJobResponse:
    def test_queued(self, mock_job_response):
        job = JobResponse(**mock_job_response)
        assert job.status == "queued"
        assert job.queue_position == 1
        assert job.result_url is None

    def test_completed(self, mock_completed_job):
        job = JobResponse(**mock_completed_job)
        assert job.status == "completed"
        assert job.result_url is not None


class TestOperationInfo:
    def test_fields(self):
        op = OperationInfo(
            description="Test op",
            allowed_targets=["whole_image", "prominent_person"],
            default_target="whole_image",
            credit_cost=2,
        )
        assert op.credit_cost == 2
        assert len(op.allowed_targets) == 2
