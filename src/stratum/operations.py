"""Operation constants mirroring the Stratum API operations registry."""

from __future__ import annotations

from .exceptions import ValidationError
from .models import OperationInfo

OPERATIONS: dict[str, OperationInfo] = {
    # ── Embeddings ────────────────────────────────────────────────
    "embed_clip_vit_b_32": OperationInfo(
        description="CLIP ViT-B/32 semantic embedding (512-d).",
        allowed_targets=["whole_image", "prominent_person", "prominent_face"],
        default_target="whole_image",
        credit_cost=1,
    ),
    "embed_dino_v2": OperationInfo(
        description="DINOv2 visual embedding (768-d).",
        allowed_targets=["whole_image", "prominent_person", "prominent_face"],
        default_target="whole_image",
        credit_cost=1,
    ),
    "embed_dino_v3_cls": OperationInfo(
        description="DINOv3 ViT-L/16 global CLS token (1024-d).",
        allowed_targets=["whole_image"],
        default_target="whole_image",
        credit_cost=1,
    ),
    "embed_dino_v3_full": OperationInfo(
        description="DINOv3 ViT-L/16 CLS + spatial patch tokens.",
        allowed_targets=["whole_image"],
        default_target="whole_image",
        credit_cost=2,
    ),
    # ── Detection ─────────────────────────────────────────────────
    "detect_bounding_box": OperationInfo(
        description="Detect bounding box for a specified target.",
        allowed_targets=["prominent_person", "prominent_face"],
        default_target="prominent_person",
        credit_cost=1,
    ),
    # ── Dense prediction ──────────────────────────────────────────
    "extract_pose": OperationInfo(
        description="DWPose-L whole-body keypoints (133 points).",
        allowed_targets=["prominent_person", "prominent_face"],
        default_target="prominent_person",
        credit_cost=3,
    ),
    "segment_body": OperationInfo(
        description="Sapiens-1B 28-class body-part segmentation.",
        allowed_targets=["prominent_person", "prominent_face"],
        default_target="prominent_person",
        credit_cost=5,
    ),
    "estimate_depth": OperationInfo(
        description="Sapiens-1B monocular depth estimation.",
        allowed_targets=["prominent_person", "prominent_face"],
        default_target="prominent_person",
        credit_cost=5,
    ),
    "estimate_normals": OperationInfo(
        description="Sapiens-1B surface normal estimation (XYZ).",
        allowed_targets=["prominent_person", "prominent_face"],
        default_target="prominent_person",
        credit_cost=5,
    ),
    # ── Text / Captioning ─────────────────────────────────────────
    "caption_image": OperationInfo(
        description="Dense objective image caption via Ollama.",
        allowed_targets=["whole_image", "prominent_person", "prominent_face"],
        default_target="whole_image",
        credit_cost=2,
    ),
    "encode_t5": OperationInfo(
        description="T5-Large encoding of image caption (512×1024).",
        allowed_targets=["whole_image"],
        default_target="whole_image",
        credit_cost=2,
    ),
}


def validate_operations(operation_types: list[str]) -> None:
    """Raise ValidationError if any operation type is unknown."""
    unknown = [op for op in operation_types if op not in OPERATIONS]
    if unknown:
        raise ValidationError(
            f"Unknown operations: {', '.join(unknown)}. "
            f"Valid operations: {', '.join(sorted(OPERATIONS))}",
            status_code=400,
        )


def estimate_credits(operation_types: list[str]) -> int:
    """Estimate total credit cost for a set of operations."""
    validate_operations(operation_types)
    return sum(OPERATIONS[op].credit_cost for op in operation_types)


def list_operations() -> dict[str, OperationInfo]:
    """Return all available operations."""
    return dict(OPERATIONS)
