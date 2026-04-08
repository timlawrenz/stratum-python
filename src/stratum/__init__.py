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
    JobResponse,
    LaneStatus,
    OperationInfo,
    SystemStatusResponse,
    TaskRequest,
)
from .operations import OPERATIONS, estimate_credits, list_operations, validate_operations
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
    "LaneStatus",
    "OperationInfo",
    "SystemStatusResponse",
    "TaskRequest",
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
    # Operations
    "OPERATIONS",
    "estimate_credits",
    "list_operations",
    "validate_operations",
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
