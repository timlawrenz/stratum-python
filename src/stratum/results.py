"""Typed result parsers for Stratum API operation outputs."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ── Result types ──────────────────────────────────────────────────────


@dataclass
class EmbeddingResult:
    """Vector embedding result (CLIP, DINOv2, DINOv3 CLS)."""

    embedding: list[float]
    dimensions: int = 0

    def __post_init__(self) -> None:
        self.dimensions = len(self.embedding)

    def to_numpy(self) -> Any:
        """Convert to numpy array. Requires numpy."""
        if not HAS_NUMPY:
            raise ImportError("Install numpy: pip install stratum[numpy]")
        return np.array(self.embedding, dtype=np.float32)


@dataclass
class DINOv3FullResult:
    """DINOv3 full result: CLS token + spatial patch embeddings."""

    cls: list[float]
    patches: list[list[float]]
    cls_dimensions: int = 0
    num_patches: int = 0

    def __post_init__(self) -> None:
        self.cls_dimensions = len(self.cls)
        self.num_patches = len(self.patches)


@dataclass
class BBox:
    """Bounding box coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DetectionResult:
    """Person or face bounding box detection."""

    detected: bool
    bbox: BBox | None
    confidence: float


@dataclass
class Keypoint:
    """Single pose keypoint with confidence."""

    x: float
    y: float
    confidence: float


@dataclass
class PoseResult:
    """DWPose whole-body keypoint extraction (133 points)."""

    keypoints: list[Keypoint]
    num_keypoints: int = 0

    def __post_init__(self) -> None:
        self.num_keypoints = len(self.keypoints)


@dataclass
class SegmentationResult:
    """Body-part segmentation mask."""

    mask_base64: str
    shape: tuple[int, int]
    classes: list[str]
    num_classes: int = 0

    def __post_init__(self) -> None:
        self.num_classes = len(self.classes)

    def decode_mask(self) -> bytes:
        """Decode base64 mask to raw bytes."""
        return base64.b64decode(self.mask_base64)

    def to_numpy(self) -> Any:
        """Decode mask to numpy array. Requires numpy."""
        if not HAS_NUMPY:
            raise ImportError("Install numpy: pip install stratum[numpy]")
        raw = self.decode_mask()
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.shape)


@dataclass
class DepthResult:
    """Monocular depth estimation output."""

    depth_base64: str
    shape: tuple[int, int]
    dtype: str = "float16"

    def decode(self) -> bytes:
        """Decode base64 depth map to raw bytes."""
        return base64.b64decode(self.depth_base64)

    def to_numpy(self) -> Any:
        """Decode to numpy array. Requires numpy."""
        if not HAS_NUMPY:
            raise ImportError("Install numpy: pip install stratum[numpy]")
        raw = self.decode()
        np_dtype = np.dtype(self.dtype)
        return np.frombuffer(raw, dtype=np_dtype).reshape(self.shape)


@dataclass
class NormalsResult:
    """Surface normal estimation output (XYZ per pixel)."""

    normals_base64: str
    shape: tuple[int, int, int]  # (H, W, 3)
    dtype: str = "float16"

    def decode(self) -> bytes:
        """Decode base64 normals map to raw bytes."""
        return base64.b64decode(self.normals_base64)

    def to_numpy(self) -> Any:
        """Decode to numpy array (H, W, 3). Requires numpy."""
        if not HAS_NUMPY:
            raise ImportError("Install numpy: pip install stratum[numpy]")
        raw = self.decode()
        np_dtype = np.dtype(self.dtype)
        return np.frombuffer(raw, dtype=np_dtype).reshape(self.shape)


@dataclass
class CaptionResult:
    """Image caption from Ollama vision model."""

    text: str


@dataclass
class T5Result:
    """T5-Large text encoding."""

    hidden_base64: str
    hidden_shape: tuple[int, int]  # (seq_len, 1024)
    mask_base64: str
    mask_shape: tuple[int]  # (seq_len,)

    def to_numpy(self) -> tuple[Any, Any]:
        """Decode hidden states and attention mask. Requires numpy.

        Returns:
            (hidden_states, attention_mask) as numpy arrays.
        """
        if not HAS_NUMPY:
            raise ImportError("Install numpy: pip install stratum[numpy]")
        hidden = np.frombuffer(
            base64.b64decode(self.hidden_base64), dtype=np.float32
        ).reshape(self.hidden_shape)
        mask = np.frombuffer(
            base64.b64decode(self.mask_base64), dtype=np.int64
        ).reshape(self.mask_shape)
        return hidden, mask


# ── Task result wrapper ───────────────────────────────────────────────


@dataclass
class TaskResult:
    """Generic wrapper for a single task result."""

    status: str
    data: Any = None
    error_message: str | None = None
    operation_type: str = ""
    parsed: (
        EmbeddingResult
        | DINOv3FullResult
        | DetectionResult
        | PoseResult
        | SegmentationResult
        | DepthResult
        | NormalsResult
        | CaptionResult
        | T5Result
        | None
    ) = None


# ── Parsed results container ─────────────────────────────────────────


@dataclass
class JobResults:
    """Container for all results from a completed job.

    Access results by operation_id as dict keys or as attributes.
    """

    _results: dict[str, TaskResult] = field(default_factory=dict)

    def __getitem__(self, key: str) -> TaskResult:
        return self._results[key]

    def __getattr__(self, name: str) -> TaskResult:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._results[name]
        except KeyError:
            raise AttributeError(f"No result for operation '{name}'") from None

    def __contains__(self, key: str) -> bool:
        return key in self._results

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def keys(self):
        return self._results.keys()

    def values(self):
        return self._results.values()

    def items(self):
        return self._results.items()


# ── Parser registry ──────────────────────────────────────────────────

_EMBEDDING_OPS = {"embed_clip_vit_b_32", "embed_dino_v2", "embed_dino_v3_cls"}


def _parse_result(operation_type: str, data: Any) -> Any:
    """Parse raw result data into a typed result object."""
    if data is None:
        return None

    if operation_type in _EMBEDDING_OPS:
        if isinstance(data, list):
            return EmbeddingResult(embedding=data)

    if operation_type == "embed_dino_v3_full":
        if isinstance(data, dict):
            return DINOv3FullResult(
                cls=data.get("cls", []),
                patches=data.get("patches", []),
            )

    if operation_type == "detect_bounding_box":
        if isinstance(data, dict):
            bbox = None
            if data.get("bbox"):
                coords = data["bbox"]
                bbox = BBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])
            return DetectionResult(
                detected=data.get("detected", False),
                bbox=bbox,
                confidence=data.get("confidence", 0.0),
            )

    if operation_type == "extract_pose":
        if isinstance(data, list):
            keypoints = [Keypoint(x=kp[0], y=kp[1], confidence=kp[2]) for kp in data]
            return PoseResult(keypoints=keypoints)

    if operation_type == "segment_body":
        if isinstance(data, dict):
            return SegmentationResult(
                mask_base64=data.get("mask_base64", ""),
                shape=tuple(data.get("mask_shape", [0, 0])),
                classes=data.get("classes", []),
            )

    if operation_type == "estimate_depth":
        if isinstance(data, dict):
            return DepthResult(
                depth_base64=data.get("depth_base64", ""),
                shape=tuple(data.get("shape", [0, 0])),
                dtype=data.get("dtype", "float16"),
            )

    if operation_type == "estimate_normals":
        if isinstance(data, dict):
            return NormalsResult(
                normals_base64=data.get("normals_base64", ""),
                shape=tuple(data.get("shape", [0, 0, 0])),
                dtype=data.get("dtype", "float16"),
            )

    if operation_type == "caption_image":
        if isinstance(data, dict):
            return CaptionResult(text=data.get("caption", ""))

    if operation_type == "encode_t5":
        if isinstance(data, dict):
            return T5Result(
                hidden_base64=data.get("hidden_base64", ""),
                hidden_shape=tuple(data.get("hidden_shape", [0, 0])),
                mask_base64=data.get("mask_base64", ""),
                mask_shape=tuple(data.get("mask_shape", [0])),
            )

    return data


def parse_job_results(
    raw: dict[str, Any], task_type_map: dict[str, str] | None = None
) -> JobResults:
    """Parse raw job result JSON into typed JobResults.

    Args:
        raw: Dict mapping operation_id → {status, data, error_message}.
        task_type_map: Optional mapping of operation_id → operation type.
            If not provided, the operation_id is used as the type.
    """
    results: dict[str, TaskResult] = {}
    for op_id, value in raw.items():
        if isinstance(value, dict) and "status" in value:
            op_type = (task_type_map or {}).get(op_id, op_id)
            parsed = None
            if value.get("status") == "success" and value.get("data") is not None:
                parsed = _parse_result(op_type, value["data"])
            results[op_id] = TaskResult(
                status=value["status"],
                data=value.get("data"),
                error_message=value.get("error_message"),
                operation_type=op_type,
                parsed=parsed,
            )
    return JobResults(_results=results)
