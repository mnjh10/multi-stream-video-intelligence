"""Tests for Stage 4: YOLO11 Object Detection & Bounding-Box Validation."""

import json
import math
from pathlib import Path

import pytest
import torch
from ultralytics import YOLO

from backend.cv.detection import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_MODEL_NAME,
    DetectionConfig,
    validate_and_clip_bbox,
    validate_detection_record,
)


# ============================================================================
# 1. CONFIGURATION TESTS
# ============================================================================


def test_detection_config_defaults():
    """Verify default detection configuration parameters."""
    cfg = DetectionConfig()
    assert cfg.model_name == DEFAULT_MODEL_NAME
    assert cfg.conf_threshold == DEFAULT_CONF_THRESHOLD
    assert cfg.batch_size == 32
    assert cfg.clip_boxes is True
    assert cfg.min_box_size == 1.0


def test_detection_config_custom_values():
    """Verify custom detection configuration values."""
    cfg = DetectionConfig(conf_threshold=0.5, batch_size=16, min_box_size=2.0)
    assert cfg.conf_threshold == 0.5
    assert cfg.batch_size == 16
    assert cfg.min_box_size == 2.0


@pytest.mark.parametrize("invalid_conf", [0.0, -0.1, 1.5, -10.0])
def test_detection_config_rejects_invalid_conf(invalid_conf):
    """Verify out-of-range confidence thresholds are rejected."""
    with pytest.raises(ValueError, match="conf_threshold must be in"):
        DetectionConfig(conf_threshold=invalid_conf)


@pytest.mark.parametrize("invalid_batch", [0, -1, -10])
def test_detection_config_rejects_invalid_batch_size(invalid_batch):
    """Verify non-positive batch sizes are rejected."""
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        DetectionConfig(batch_size=invalid_batch)


# ============================================================================
# 2. BOUNDING-BOX VALIDATION & CLIPPING TESTS
# ============================================================================


def test_validate_bbox_valid_box():
    """Verify valid bounding box within image boundaries passes."""
    is_valid, clipped, err = validate_and_clip_bbox(
        bbox=[100.0, 50.0, 200.0, 150.0],
        image_width=1920,
        image_height=1080,
    )
    assert is_valid is True
    assert clipped == [100.0, 50.0, 200.0, 150.0]
    assert err is None


def test_validate_bbox_clipping_policy():
    """Verify that bounding boxes extending outside boundaries are clipped cleanly."""
    is_valid, clipped, err = validate_and_clip_bbox(
        bbox=[-10.5, -5.0, 1950.0, 1100.0],
        image_width=1920,
        image_height=1080,
        clip=True,
    )
    assert is_valid is True
    assert clipped == [0.0, 0.0, 1920.0, 1080.0]


def test_validate_bbox_degenerate_zero_area_rejected():
    """Verify degenerate box with zero width or zero height is rejected."""
    # Zero width
    is_valid, _, err = validate_and_clip_bbox(
        bbox=[100.0, 50.0, 100.0, 150.0],
        image_width=1920,
        image_height=1080,
    )
    assert is_valid is False
    assert "below minimum size" in err

    # Inverted coordinates
    is_valid, _, err = validate_and_clip_bbox(
        bbox=[200.0, 50.0, 100.0, 150.0],
        image_width=1920,
        image_height=1080,
    )
    assert is_valid is False


@pytest.mark.parametrize("bad_coord", [float("nan"), float("inf"), float("-inf")])
def test_validate_bbox_nan_inf_rejected(bad_coord):
    """Verify non-finite coordinates are strictly rejected."""
    is_valid, _, err = validate_and_clip_bbox(
        bbox=[0.0, 0.0, bad_coord, 100.0],
        image_width=1920,
        image_height=1080,
    )
    assert is_valid is False
    assert "finite number" in err


def test_validate_bbox_wrong_element_count():
    """Verify bounding boxes with != 4 coordinates are rejected."""
    is_valid, _, err = validate_and_clip_bbox(
        bbox=[10.0, 20.0, 30.0],
        image_width=1920,
        image_height=1080,
    )
    assert is_valid is False
    assert "exactly 4 coordinates" in err


# ============================================================================
# 3. DETECTION RECORD SCHEMA VALIDATION TESTS
# ============================================================================


def test_validate_detection_record_valid():
    """Verify schema validator accepts complete, well-formed detection records."""
    rec = {
        "detection_id": "det_000001",
        "camera_id": "cam_01",
        "frame_index": 30,
        "timestamp": 3.0,
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.895,
        "bbox": [100.0, 200.0, 300.0, 400.0],
        "frame_path": "data/frames/cam_01/frame_000030.jpg",
        "source_video": "data/videos/cam_01/vdo.avi",
    }
    is_valid, err = validate_detection_record(rec, image_width=1920, image_height=1080)
    assert is_valid is True
    assert err is None


def test_validate_detection_record_missing_key():
    """Verify schema validator flags missing keys."""
    rec = {
        "detection_id": "det_000001",
        "camera_id": "cam_01",
        "frame_index": 30,
        # missing timestamp
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.895,
        "bbox": [100.0, 200.0, 300.0, 400.0],
        "frame_path": "data/frames/cam_01/frame_000030.jpg",
        "source_video": "data/videos/cam_01/vdo.avi",
    }
    is_valid, err = validate_detection_record(rec, image_width=1920, image_height=1080)
    assert is_valid is False
    assert "Missing required key: 'timestamp'" in err


# ============================================================================
# 4. REAL INFERENCE & SMOKE TEST
# ============================================================================


def test_yolo11_inference_on_real_frame():
    """Verify YOLO11 model runs inference on a real Stage 3 frame and returns detections."""
    frame_path = Path("data/frames/cam_01/frame_000000.jpg")
    if not frame_path.exists():
        pytest.skip("data/frames/cam_01/frame_000000.jpg does not exist")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt")

    results = model(str(frame_path), conf=0.25, device=device, verbose=False)
    assert len(results) == 1
    res = results[0]

    assert res.boxes is not None
    assert len(res.boxes) > 0, "Expected at least 1 detection on test frame"

    # Verify first box properties
    box = res.boxes[0]
    cls_id = int(box.cls[0].item())
    cls_name = model.names[cls_id]
    conf = float(box.conf[0].item())
    xyxy = box.xyxy[0].tolist()

    assert cls_id >= 0
    assert isinstance(cls_name, str) and len(cls_name) > 0
    assert 0.0 <= conf <= 1.0

    # Verify bbox satisfies native image resolution
    orig_h, orig_w = res.orig_shape
    assert orig_w == 1920
    assert orig_h == 1080

    is_valid, clipped, _ = validate_and_clip_bbox(xyxy, orig_w, orig_h)
    assert is_valid is True
    assert clipped[0] < clipped[2]
    assert clipped[1] < clipped[3]


# ============================================================================
# 5. DETERMINISM TEST
# ============================================================================


def test_yolo11_inference_determinism():
    """Verify running inference twice on the same input yields identical detections."""
    frame_path = Path("data/frames/cam_01/frame_000000.jpg")
    if not frame_path.exists():
        pytest.skip("data/frames/cam_01/frame_000000.jpg does not exist")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n.pt")

    res1 = model(str(frame_path), conf=0.25, device=device, verbose=False)[0]
    res2 = model(str(frame_path), conf=0.25, device=device, verbose=False)[0]

    assert len(res1.boxes) == len(res2.boxes)
    for b1, b2 in zip(res1.boxes, res2.boxes):
        assert int(b1.cls[0].item()) == int(b2.cls[0].item())
        # Confidences equal to 3 decimal places
        assert round(float(b1.conf[0].item()), 3) == round(float(b2.conf[0].item()), 3)
        # Bbox coordinates equal to 1 decimal place
        xy1 = [round(x, 1) for x in b1.xyxy[0].tolist()]
        xy2 = [round(x, 1) for x in b2.xyxy[0].tolist()]
        assert xy1 == xy2


# ============================================================================
# 6. FULL DATASET DETECTION VERIFICATION TESTS
# ============================================================================


def test_stage4_detection_manifest_structure():
    """Verify data/detections/detection_manifest.json exists and adheres to contract."""
    manifest_path = Path("data/detections/detection_manifest.json")
    if not manifest_path.exists():
        pytest.skip("data/detections/detection_manifest.json does not exist")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == "1.0"
    assert manifest["model_name"] == "yolo11n.pt"
    assert manifest["conf_threshold"] == 0.25
    assert manifest["total_input_frames"] == 7287
    assert manifest["processed_frames"] == 7287
    assert manifest["failed_frames"] == 0
    assert manifest["total_detections"] == 68906

    # Verify camera coverage across all 11 cameras
    cam_counts = manifest["detections_per_camera"]
    assert len(cam_counts) == 11
    expected_cams = [f"cam_{i:02d}" for i in range(1, 12)]
    for cam_id in expected_cams:
        assert cam_id in cam_counts
        assert cam_counts[cam_id] > 0

    assert sum(cam_counts.values()) == 68906

    # Verify authoritative frames per camera matches Stage 3 exactly
    expected_frames_per_cam = {
        "cam_01": 587,
        "cam_02": 633,
        "cam_03": 599,
        "cam_04": 633,
        "cam_05": 633,
        "cam_06": 643,
        "cam_07": 684,
        "cam_08": 727,
        "cam_09": 725,
        "cam_10": 700,
        "cam_11": 723,
    }
    if "frames_per_camera" in manifest:
        f_counts = manifest["frames_per_camera"]
        assert f_counts == expected_frames_per_cam
        assert sum(f_counts.values()) == 7287

    # Verify class distribution
    class_dist = manifest["class_distribution"]
    assert sum(class_dist.values()) == 68906
    assert class_dist["car"] > 50000

    # Verify confidence stats
    conf_stats = manifest["confidence_statistics"]
    assert conf_stats["min"] >= 0.25
    assert conf_stats["max"] <= 1.0
    assert 0.5 <= conf_stats["mean"] <= 0.6


def test_stage4_detections_jsonl_records():
    """Verify data/detections/detections.jsonl contains 68,906 valid, strictly bounded records."""
    jsonl_path = Path("data/detections/detections.jsonl")
    if not jsonl_path.exists():
        pytest.skip("data/detections/detections.jsonl does not exist")

    # Load authoritative frame index map for timestamp cross-verification
    frame_index_path = Path("data/frames/frame_index.jsonl")
    frame_map = {}
    if frame_index_path.exists():
        with open(frame_index_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                frame_map[(rec["camera_id"], rec["frame_index"])] = rec

    count = 0
    seen_cameras = set()
    first_id = None
    last_id = None
    frames_with_detections = {f"cam_{i:02d}": set() for i in range(1, 12)}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            count += 1
            rec = json.loads(line)
            if idx == 0:
                first_id = rec["detection_id"]
            last_id = rec["detection_id"]

            cam_id = rec["camera_id"]
            seen_cameras.add(cam_id)
            f_idx = rec["frame_index"]
            frames_with_detections[cam_id].add(f_idx)
            bbox = rec["bbox"]

            # Bounds & validity
            assert len(bbox) == 4
            assert bbox[0] < bbox[2], f"x1 >= x2 on line {idx+1}"
            assert bbox[1] < bbox[3], f"y1 >= y2 on line {idx+1}"
            assert bbox[0] >= 0.0
            assert bbox[1] >= 0.0

            assert rec["confidence"] >= 0.25

            # Scope boundary check: no tracking IDs allowed in Stage 4
            assert "track_id" not in rec, "track_id found in Stage 4 detection record!"

            # Verify every detection references a real Stage 3 frame
            assert (cam_id, f_idx) in frame_map, f"Detection references invalid frame: ({cam_id}, {f_idx})"

            # Spot-check timestamp cross-validation with frame_index.jsonl
            if idx % 1000 == 0:
                expected_ts = frame_map[(cam_id, f_idx)]["timestamp"]
                assert abs(rec["timestamp"] - expected_ts) < 1e-4

    assert count == 68906
    assert first_id == "det_000001"
    assert last_id == f"det_{count:06d}"
    assert len(seen_cameras) == 11

    # Verify 100% frame coverage per camera matching Stage 3 counts
    expected_frames_per_cam = {
        "cam_01": 587,
        "cam_02": 633,
        "cam_03": 599,
        "cam_04": 633,
        "cam_05": 633,
        "cam_06": 643,
        "cam_07": 684,
        "cam_08": 727,
        "cam_09": 725,
        "cam_10": 700,
        "cam_11": 723,
    }
    for cam_id, expected_n in expected_frames_per_cam.items():
        assert len(frames_with_detections[cam_id]) == expected_n, (
            f"Coverage mismatch for {cam_id}: got {len(frames_with_detections[cam_id])}, expected {expected_n}"
        )


