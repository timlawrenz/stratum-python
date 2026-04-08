"""Tests for result parsing."""

from __future__ import annotations

from stratum.results import (
    BBox,
    CaptionResult,
    DetectionResult,
    EmbeddingResult,
    Keypoint,
    PoseResult,
    SegmentationResult,
    parse_job_results,
)


class TestEmbeddingResult:
    def test_dimensions(self):
        r = EmbeddingResult(embedding=[0.1, 0.2, 0.3])
        assert r.dimensions == 3

    def test_empty(self):
        r = EmbeddingResult(embedding=[])
        assert r.dimensions == 0


class TestBBox:
    def test_properties(self):
        bbox = BBox(x1=10, y1=20, x2=110, y2=220)
        assert bbox.width == 100
        assert bbox.height == 200
        assert bbox.center == (60.0, 120.0)


class TestDetectionResult:
    def test_detected(self):
        r = DetectionResult(
            detected=True,
            bbox=BBox(x1=10, y1=20, x2=110, y2=220),
            confidence=0.95,
        )
        assert r.detected is True
        assert r.bbox is not None
        assert r.bbox.width == 100

    def test_not_detected(self):
        r = DetectionResult(detected=False, bbox=None, confidence=0.0)
        assert r.detected is False
        assert r.bbox is None


class TestPoseResult:
    def test_keypoints(self):
        kps = [Keypoint(x=0.5, y=0.3, confidence=0.99)] * 133
        r = PoseResult(keypoints=kps)
        assert r.num_keypoints == 133
        assert r.keypoints[0].x == 0.5


class TestParseJobResults:
    def test_clip_embedding(self, mock_clip_result):
        results = parse_job_results(mock_clip_result)
        assert "embed_clip_vit_b_32" in results
        task = results["embed_clip_vit_b_32"]
        assert task.status == "success"
        assert isinstance(task.parsed, EmbeddingResult)
        assert task.parsed.dimensions == 512

    def test_detection(self, mock_detection_result):
        results = parse_job_results(mock_detection_result)
        task = results["detect_bounding_box"]
        assert isinstance(task.parsed, DetectionResult)
        assert task.parsed.detected is True
        assert task.parsed.bbox.x1 == 100.0

    def test_pose(self, mock_pose_result):
        results = parse_job_results(mock_pose_result)
        task = results["extract_pose"]
        assert isinstance(task.parsed, PoseResult)
        assert task.parsed.num_keypoints == 133

    def test_multiple_results(self, mock_full_results):
        results = parse_job_results(mock_full_results)
        assert len(results) == 4
        assert "embed_clip_vit_b_32" in results
        assert "detect_bounding_box" in results
        assert "segment_body" in results
        assert "caption_image" in results

        # Check caption
        caption = results["caption_image"]
        assert isinstance(caption.parsed, CaptionResult)
        assert caption.parsed.text == "A person standing outdoors."

        # Check segmentation
        seg = results["segment_body"]
        assert isinstance(seg.parsed, SegmentationResult)
        assert seg.parsed.num_classes == 2

    def test_error_result(self):
        raw = {
            "my_op": {
                "status": "error",
                "data": None,
                "error_message": "CUDA OOM",
            }
        }
        results = parse_job_results(raw)
        task = results["my_op"]
        assert task.status == "error"
        assert task.error_message == "CUDA OOM"
        assert task.parsed is None

    def test_with_type_map(self):
        raw = {
            "my_clip": {
                "status": "success",
                "data": [0.1] * 512,
            }
        }
        results = parse_job_results(raw, task_type_map={"my_clip": "embed_clip_vit_b_32"})
        task = results["my_clip"]
        assert isinstance(task.parsed, EmbeddingResult)


class TestJobResults:
    def test_dict_access(self, mock_clip_result):
        results = parse_job_results(mock_clip_result)
        assert results["embed_clip_vit_b_32"].status == "success"

    def test_attr_access(self, mock_clip_result):
        results = parse_job_results(mock_clip_result)
        assert results.embed_clip_vit_b_32.status == "success"

    def test_contains(self, mock_clip_result):
        results = parse_job_results(mock_clip_result)
        assert "embed_clip_vit_b_32" in results
        assert "nonexistent" not in results

    def test_len(self, mock_full_results):
        results = parse_job_results(mock_full_results)
        assert len(results) == 4

    def test_iter(self, mock_full_results):
        results = parse_job_results(mock_full_results)
        keys = list(results)
        assert "embed_clip_vit_b_32" in keys
        assert len(keys) == 4
