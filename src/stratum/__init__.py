"""Stratum — Python SDK for the Stratum image enrichment API."""

from .async_client import AsyncStratumClient
from .client import StratumClient
from .exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    JobFailedError,
    JobTimeoutError,
    RateLimitError,
    ServerError,
    StratumError,
    ValidationError,
)
from .models import (
    AnalyzeImageRequest,
    AvailableOperationsResponse,
    BalanceResponse,
    BatchRequest,
    BatchResponse,
    BatchStatusResponse,
    FaceTasks,
    JobResponse,
    JobsPage,
    JobStatus,
    LaneStatus,
    OperationInfo,
    PersonTasks,
    SystemStatusResponse,
    WholeImageTasks,
)
from .operations import estimate_credits, list_operations
from .results import (
    BBox,
    CaptionResult,
    DepthResult,
    DetectionResult,
    DINOv3FullResult,
    EmbeddingResult,
    JobResults,
    Keypoint,
    NormalsResult,
    PoseResult,
    SectionResults,
    SegmentationResult,
    T5Result,
    TaskResult,
)
from .webhook import verify_signature

__all__ = [
    # Clients
    "StratumClient",
    "AsyncStratumClient",
    # Models
    "AnalyzeImageRequest",
    "AvailableOperationsResponse",
    "BalanceResponse",
    "BatchRequest",
    "BatchResponse",
    "BatchStatusResponse",
    "JobResponse",
    "JobsPage",
    "JobStatus",
    "LaneStatus",
    "OperationInfo",
    "SystemStatusResponse",
    "WholeImageTasks",
    "PersonTasks",
    "FaceTasks",
    # Results
    "BBox",
    "CaptionResult",
    "DepthResult",
    "DetectionResult",
    "DINOv3FullResult",
    "EmbeddingResult",
    "JobResults",
    "Keypoint",
    "NormalsResult",
    "PoseResult",
    "SegmentationResult",
    "T5Result",
    "TaskResult",
    "SectionResults",
    # Operations
    "estimate_credits",
    "list_operations",
    # Exceptions
    "AuthenticationError",
    "InsufficientCreditsError",
    "JobFailedError",
    "JobTimeoutError",
    "RateLimitError",
    "ServerError",
    "StratumError",
    "ValidationError",
    # Webhook
    "verify_signature",
]
