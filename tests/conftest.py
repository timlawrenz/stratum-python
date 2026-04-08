"""Test fixtures for Stratum SDK tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest


@pytest.fixture
def api_key() -> str:
    return "sk_test_abc123"


@pytest.fixture
def base_url() -> str:
    return "https://test.stratum.api"


@pytest.fixture
def job_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def mock_job_response(job_id: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "queued",
        "sla_lane": "within_minutes",
        "queue_position": 1,
        "estimated_wait_seconds": 10.0,
        "result_url": None,
        "error_message": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
    }


@pytest.fixture
def mock_completed_job(job_id: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "completed",
        "sla_lane": "within_minutes",
        "queue_position": None,
        "estimated_wait_seconds": None,
        "result_url": "https://r2.example.com/results/test.json",
        "error_message": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def mock_clip_result() -> dict[str, Any]:
    return {
        "embed_clip_vit_b_32": {
            "status": "success",
            "data": [0.1] * 512,
        }
    }


@pytest.fixture
def mock_detection_result() -> dict[str, Any]:
    return {
        "detect_bounding_box": {
            "status": "success",
            "data": {
                "detected": True,
                "bbox": [100.0, 50.0, 500.0, 800.0],
                "confidence": 0.95,
            },
        }
    }


@pytest.fixture
def mock_pose_result() -> dict[str, Any]:
    return {
        "extract_pose": {
            "status": "success",
            "data": [[0.5, 0.3, 0.99]] * 133,
        }
    }


@pytest.fixture
def mock_full_results() -> dict[str, Any]:
    return {
        "embed_clip_vit_b_32": {
            "status": "success",
            "data": [0.1] * 512,
        },
        "detect_bounding_box": {
            "status": "success",
            "data": {
                "detected": True,
                "bbox": [100.0, 50.0, 500.0, 800.0],
                "confidence": 0.95,
            },
        },
        "segment_body": {
            "status": "success",
            "data": {
                "mask_base64": "AAAA",
                "mask_shape": [2, 2],
                "classes": ["background", "head"],
            },
        },
        "caption_image": {
            "status": "success",
            "data": {
                "caption": "A person standing outdoors.",
            },
        },
    }
