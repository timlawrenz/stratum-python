"""Tests for Pydantic models."""

from __future__ import annotations

import pytest

from stratum.models import (
    AnalyzeImageRequest,
    FaceTasks,
    JobResponse,
    OperationInfo,
    PersonTasks,
    WholeImageTasks,
)


class TestAnalyzeImageRequest:
    def test_minimal(self):
        req = AnalyzeImageRequest(
            image_url="https://example.com/img.jpg",
            whole_image=WholeImageTasks(clip=True),
        )
        assert req.callback_url is None

    def test_with_callback(self):
        req = AnalyzeImageRequest(
            image_url="https://example.com/img.jpg",
            whole_image=WholeImageTasks(clip=True),
            callback_url="https://webhook.site/test",
        )
        assert req.callback_url == "https://webhook.site/test"

    def test_image_url_or_base64(self):
        msg = "exactly one of image_url or image_base64 must be provided"
        with pytest.raises(ValueError, match=msg):
            AnalyzeImageRequest(
                whole_image=WholeImageTasks(clip=True),
            )

        with pytest.raises(ValueError, match=msg):
            AnalyzeImageRequest(
                image_url="http://a.com/a.jpg",
                image_base64="aGVsbG8=",
                whole_image=WholeImageTasks(clip=True),
            )

        req = AnalyzeImageRequest(
            image_base64="aGVsbG8=",
            whole_image=WholeImageTasks(clip=True),
        )
        assert req.image_base64 == "aGVsbG8="

    def test_face_requires_person(self):
        with pytest.raises(ValueError, match="prominent_face requires prominent_person"):
            AnalyzeImageRequest(
                image_url="https://example.com/img.jpg",
                prominent_face=FaceTasks(clip=True),
            )

    def test_t5_requires_caption(self):
        with pytest.raises(ValueError, match="t5 requires caption"):
            AnalyzeImageRequest(
                image_url="https://example.com/img.jpg",
                whole_image=WholeImageTasks(t5=True),
            )

    def test_person_depth_requires_seg(self):
        with pytest.raises(ValueError, match="depth and normal require seg in prominent_person"):
            AnalyzeImageRequest(
                image_url="https://example.com/img.jpg",
                prominent_person=PersonTasks(depth=True),
            )

    def test_face_depth_requires_seg(self):
        with pytest.raises(ValueError, match="depth and normal require seg in prominent_face"):
            AnalyzeImageRequest(
                image_url="https://example.com/img.jpg",
                prominent_person=PersonTasks(seg=True),
                prominent_face=FaceTasks(depth=True),
            )

    def test_at_least_one_section(self):
        with pytest.raises(ValueError, match="at least one section"):
            AnalyzeImageRequest(image_url="https://example.com/img.jpg")


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
            credit_cost=2,
        )
        assert op.credit_cost == 2
