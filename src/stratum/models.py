"""Pydantic models mirroring the Stratum API schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

# ── Requests ──────────────────────────────────────────────────────────


class TaskRequest(BaseModel):
    """A single operation to perform on an image."""

    operation_id: str = ""
    type: str
    params: dict = Field(default_factory=dict)


class AnalyzeImageRequest(BaseModel):
    """Submit an image for enrichment."""

    image_url: str
    tasks: list[TaskRequest]
    sla_lane: str = "within_minutes"
    callback_url: str | None = None


class BatchRequest(BaseModel):
    """Submit a batch of images by R2 prefix."""

    r2_prefix: str
    tasks: list[TaskRequest]
    sla_lane: str = "within_hours"
    callback_url: str | None = None


# ── Responses ─────────────────────────────────────────────────────────


class JobResponse(BaseModel):
    """Server response for a submitted or polled job."""

    job_id: uuid.UUID
    status: str
    sla_lane: str
    queue_position: int | None = None
    estimated_wait_seconds: float | None = None
    result_url: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class BatchResponse(BaseModel):
    """Server response for a submitted batch."""

    batch_id: uuid.UUID
    total_images: int
    status: str


class BatchStatusResponse(BaseModel):
    """Batch processing progress."""

    batch_id: uuid.UUID
    total: int
    completed: int
    failed: int
    status: str


# ── Operations ────────────────────────────────────────────────────────


class OperationInfo(BaseModel):
    """Metadata for a supported operation."""

    description: str
    allowed_targets: list[str]
    default_target: str
    credit_cost: int


class AvailableOperationsResponse(BaseModel):
    """All operations the API supports."""

    operations: dict[str, OperationInfo]


# ── System status ─────────────────────────────────────────────────────


class LaneStatus(BaseModel):
    """Status of a single SLA queue lane."""

    lane: str
    depth: int
    active_workers: int
    avg_processing_time_s: float
    estimated_time_to_empty_s: float
    sla_target_s: int
    sla_healthy: bool


class SystemStatusResponse(BaseModel):
    """Overall system health."""

    lanes: list[LaneStatus]


# ── Billing ───────────────────────────────────────────────────────────


class BalanceResponse(BaseModel):
    """Current credit balance."""

    credit_balance: int
