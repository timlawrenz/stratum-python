"""Pydantic models mirroring the Stratum API schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, model_validator

# ── Requests ──────────────────────────────────────────────────────────

class WholeImageTasks(BaseModel):
    """Operations to run on the full image."""
    clip: bool = False
    dino_v2: bool = False
    dinov3_cls: bool = False
    dinov3_full: bool = False
    caption: bool = False
    t5: bool = False
    pixel: bool = False


class PersonTasks(BaseModel):
    """Operations to run on the prominent person crop."""
    clip: bool = False
    dino_v2: bool = False
    dinov3_cls: bool = False
    dinov3_full: bool = False
    caption: bool = False
    t5: bool = False
    pose: bool = False
    seg: bool = False
    depth: bool = False
    normal: bool = False


class FaceTasks(BaseModel):
    """Operations to run on the prominent face crop."""
    clip: bool = False
    dino_v2: bool = False
    dinov3_cls: bool = False
    dinov3_full: bool = False
    caption: bool = False
    t5: bool = False
    pose: bool = False
    seg: bool = False
    depth: bool = False
    normal: bool = False


class AnalyzeImageRequest(BaseModel):
    """Submit an image for enrichment."""
    image_url: str | None = None
    image_base64: str | None = None
    whole_image: WholeImageTasks | None = None
    prominent_person: PersonTasks | None = None
    prominent_face: FaceTasks | None = None
    callback_url: str | None = None

    @model_validator(mode="after")
    def _validate_sections(self) -> AnalyzeImageRequest:
        if bool(self.image_url) == bool(self.image_base64):
            raise ValueError("exactly one of image_url or image_base64 must be provided")

        if self.prominent_face is not None and self.prominent_person is None:
            raise ValueError("prominent_face requires prominent_person to also be enabled")

        for section_name in ("whole_image", "prominent_person", "prominent_face"):
            section = getattr(self, section_name)
            if (
                section is not None
                and getattr(section, "t5", False)
                and not getattr(section, "caption", False)
            ):
                raise ValueError(f"t5 requires caption in {section_name}")

        if self.prominent_person is not None:
            p = self.prominent_person
            if (p.depth or p.normal) and not p.seg:
                raise ValueError("depth and normal require seg in prominent_person")

        if self.prominent_face is not None:
            f = self.prominent_face
            if (f.depth or f.normal) and not f.seg:
                raise ValueError("depth and normal require seg in prominent_face")

        if not any([self.whole_image, self.prominent_person, self.prominent_face]):
            raise ValueError(
                "at least one section (whole_image, prominent_person, prominent_face) "
                "must be provided"
            )

        return self


class BatchRequest(BaseModel):
    """Submit a batch of images by R2 prefix."""
    r2_prefix: str
    whole_image: WholeImageTasks | None = None
    prominent_person: PersonTasks | None = None
    prominent_face: FaceTasks | None = None
    callback_url: str | None = None


# ── Responses ─────────────────────────────────────────────────────────


class JobResponse(BaseModel):
    """Server response for a submitted or polled job."""
    job_id: uuid.UUID
    status: str
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
    credit_cost: int


class AvailableOperationsResponse(BaseModel):
    """All operations the API supports."""
    whole_image: list[str]
    prominent_person: list[str]
    prominent_face: list[str]


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
