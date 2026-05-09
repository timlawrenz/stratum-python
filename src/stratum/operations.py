"""Operation constants mirroring the Stratum API operations registry."""

from __future__ import annotations

from typing import TYPE_CHECKING
from .exceptions import ValidationError
from .models import OperationInfo

if TYPE_CHECKING:
    from .models import FaceTasks, PersonTasks, WholeImageTasks, AnalyzeImageRequest


class FlagSpec:
    """Complete specification for one boolean flag inside a section."""
    def __init__(self, op_type: str, credit_cost: int, description: str = ""):
        self.op_type = op_type
        self.credit_cost = credit_cost
        self.description = description


SECTION_FLAGS: dict[str, dict[str, FlagSpec]] = {
    "whole_image": {
        "clip":        FlagSpec("embed_clip_vit_b_32",  1, "CLIP ViT-B/32"),
        "dino_v2":     FlagSpec("embed_dino_v2",        1, "DINOv2"),
        "dinov3_cls":  FlagSpec("embed_dino_v3_cls",    1, "DINOv3 CLS token"),
        "dinov3_full": FlagSpec("embed_dino_v3_full",   2, "DINOv3 cls+patches"),
        "caption":     FlagSpec("caption_image",        2, "Image caption"),
        "t5":          FlagSpec("encode_t5",            2, "T5 encoding"),
        "pixel":       FlagSpec("bucket_crop",          1, "Pixel bucket crop"),
    },
    "prominent_person": {
        "clip":        FlagSpec("embed_clip_vit_b_32",  1, "CLIP ViT-B/32"),
        "dino_v2":     FlagSpec("embed_dino_v2",        1, "DINOv2"),
        "dinov3_cls":  FlagSpec("embed_dino_v3_cls",    1, "DINOv3 CLS token"),
        "dinov3_full": FlagSpec("embed_dino_v3_full",   2, "DINOv3 cls+patches"),
        "caption":     FlagSpec("caption_image",        2, "Person caption"),
        "t5":          FlagSpec("encode_t5",            2, "T5 encoding"),
        "pose":        FlagSpec("extract_pose",         3, "Pose keypoints"),
        "seg":         FlagSpec("segment_body",         5, "Body segmentation"),
        "depth":       FlagSpec("estimate_depth",       5, "Depth map"),
        "normal":      FlagSpec("estimate_normals",     5, "Surface normals"),
    },
    "prominent_face": {
        "clip":        FlagSpec("embed_clip_vit_b_32",  1, "CLIP ViT-B/32"),
        "dino_v2":     FlagSpec("embed_dino_v2",        1, "DINOv2"),
        "dinov3_cls":  FlagSpec("embed_dino_v3_cls",    1, "DINOv3 CLS token"),
        "dinov3_full": FlagSpec("embed_dino_v3_full",   2, "DINOv3 cls+patches"),
        "caption":     FlagSpec("caption_image",        2, "Face caption"),
        "t5":          FlagSpec("encode_t5",            2, "T5 encoding"),
        "pose":        FlagSpec("extract_pose",         3, "Face pose keypoints"),
        "seg":         FlagSpec("segment_body",         5, "Face segmentation"),
        "depth":       FlagSpec("estimate_depth",       5, "Depth map"),
        "normal":      FlagSpec("estimate_normals",     5, "Surface normals"),
    },
}

DETECTION_COST = 1

_WHOLE_IMAGE_COSTS: dict[str, int] = {
    f: s.credit_cost for f, s in SECTION_FLAGS["whole_image"].items()
}
_PERSON_COSTS: dict[str, int] = {
    f: s.credit_cost for f, s in SECTION_FLAGS["prominent_person"].items()
}
_FACE_COSTS: dict[str, int] = {
    f: s.credit_cost for f, s in SECTION_FLAGS["prominent_face"].items()
}

def _section_cost(section: WholeImageTasks | PersonTasks | FaceTasks, costs: dict[str, int]) -> int:
    return sum(cost for field, cost in costs.items() if getattr(section, field, False))


def estimate_credits(req: AnalyzeImageRequest) -> int:
    """Estimate total credit cost for a section-based job request."""
    total = 0
    if req.whole_image is not None:
        total += _section_cost(req.whole_image, _WHOLE_IMAGE_COSTS)
    if req.prominent_person is not None:
        total += DETECTION_COST
        total += _section_cost(req.prominent_person, _PERSON_COSTS)
    if req.prominent_face is not None:
        total += DETECTION_COST
        total += _section_cost(req.prominent_face, _FACE_COSTS)
    return total


def list_operations() -> dict[str, list[str]]:
    """Return all available operations."""
    return {
        "whole_image": list(_WHOLE_IMAGE_COSTS),
        "prominent_person": list(_PERSON_COSTS),
        "prominent_face": list(_FACE_COSTS),
    }

def get_op_type(section: str, flag: str) -> str:
    """Get the underlying operation type for a given section flag."""
    if flag == "bbox":
        return "detect_bounding_box"
    spec = SECTION_FLAGS.get(section, {}).get(flag)
    if not spec:
        return ""
    return spec.op_type