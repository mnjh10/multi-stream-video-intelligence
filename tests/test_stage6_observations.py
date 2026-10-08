"""Tests for Stage 6: Object Crop Generation and Observation Record Construction."""

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cv.crops import (
    DEFAULT_CROP_QUALITY,
    CropConfig,
    compute_safe_crop_coordinates,
    extract_and_save_crop,
    validate_crop_file,
)
from backend.cv.observations import (
    FROZEN_OBSERVATION_KEYS,
    OBSERVATION_ID_PATTERN,
    ObservationConfig,
    format_observation_id,
    validate_observation_record,
)


# ============================================================================
# 1. CONFIGURATION TESTS
# ============================================================================


def test_observation_config_defaults():
    """Verify default observation configuration parameters."""
    cfg = ObservationConfig()
    assert cfg.jpeg_quality == DEFAULT_CROP_QUALITY
    assert cfg.num_workers == 4
    assert cfg.tracks_file == "data/tracks/tracks.jsonl"
    assert cfg.output_dir == "data/observations"
    assert cfg.crops_dir == "data/crops"


def test_observation_config_custom_values():
    """Verify custom observation configuration parameters."""
    cfg = ObservationConfig(jpeg_quality=90, num_workers=8)
    assert cfg.jpeg_quality == 90
    assert cfg.num_workers == 8


@pytest.mark.parametrize("invalid_q", [0, -5, 101, 150])
def test_observation_config_rejects_invalid_quality(invalid_q):
    """Verify out-of-range JPEG quality values are rejected."""
    with pytest.raises(ValueError, match="jpeg_quality must be in"):
        ObservationConfig(jpeg_quality=invalid_q)


@pytest.mark.parametrize("invalid_w", [0, -1, -10])
def test_observation_config_rejects_invalid_workers(invalid_w):
    """Verify non-positive worker count is rejected."""
    with pytest.raises(ValueError, match="num_workers must be >= 1"):
        ObservationConfig(num_workers=invalid_w)


# ============================================================================
# 2. CROP UTILITIES & COORDINATE TESTS
# ============================================================================


def test_compute_safe_crop_coordinates():
    """Verify conversion of float bbox to bounded safe integer coordinates."""
    x1, y1, x2, y2 = compute_safe_crop_coordinates(
        bbox=[100.2, 50.8, 200.4, 150.1],
        image_width=1920,
        image_height=1080,
    )
    assert x1 == 100
    assert y1 == 51
    assert x2 == 200
    assert y2 == 150
    assert x2 > x1
    assert y2 > y1


def test_compute_safe_crop_coordinates_boundary_clamping():
    """Verify safe crop coordinates are clamped within image dimensions."""
    x1, y1, x2, y2 = compute_safe_crop_coordinates(
        bbox=[-10.0, -5.0, 2000.0, 1100.0],
        image_width=1920,
        image_height=1080,
    )
    assert x1 == 0
    assert y1 == 0
    assert x2 == 1920
    assert y2 == 1080


def test_extract_and_save_crop_synthetic(tmp_path):
    """Verify extracting and saving a crop from an in-memory frame."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:150, 50:150] = 255  # White center square

    crop_path = tmp_path / "test_crop.jpg"
    ok, dims, err = extract_and_save_crop(frame, [50.0, 50.0, 150.0, 150.0], crop_path, jpeg_quality=95)

    assert ok is True
    assert dims == (100, 100)
    assert err is None
    assert crop_path.exists()

    # Validate saved file
    is_valid, read_dims, v_err = validate_crop_file(crop_path)
    assert is_valid is True
    assert read_dims == (100, 100)
    assert v_err is None


# ============================================================================
# 3. OBSERVATION RECORD SCHEMA & ID FORMAT TESTS
# ============================================================================


def test_format_observation_id():
    """Verify observation ID formatting convention."""
    assert format_observation_id(1) == "obs_000001"
    assert format_observation_id(42) == "obs_000042"
    assert format_observation_id(61039) == "obs_061039"


@pytest.mark.parametrize(
    "valid_id",
    [
        "obs_000001",
        "obs_012345",
        "obs_061039",
    ],
)
def test_observation_id_pattern_valid(valid_id):
    """Verify observation ID regex matches valid sequential IDs."""
    assert OBSERVATION_ID_PATTERN.match(valid_id) is not None


@pytest.mark.parametrize(
    "invalid_id",
    [
        "obs_1",
        "obs_01",
        "observation_000001",
        "obs_0000001",
        "obs_abcdef",
    ],
)
def test_observation_id_pattern_invalid(invalid_id):
    """Verify observation ID regex rejects invalid IDs."""
    assert OBSERVATION_ID_PATTERN.match(invalid_id) is None


def test_validate_observation_record_valid():
    """Verify schema validator accepts a well-formed observation record."""
    rec = {
        "observation_id": "obs_000001",
        "camera_id": "cam_01",
        "timestamp": 142.37,
        "frame_index": 4271,
        "object_id": "obj_cam01_000034",
        "object_type": "car",
        "confidence": 0.94,
        "bbox": [120.0, 80.0, 530.0, 310.0],
        "frame_path": "data/frames/cam_01/frame_004271.jpg",
        "crop_path": "data/crops/cam_01/obs_000001.jpg",
        "source_video": "data/videos/cam_01/vdo.avi",
    }
    is_valid, err = validate_observation_record(rec)
    assert is_valid is True
    assert err is None


def test_validate_observation_record_missing_key():
    """Verify schema validator rejects records missing a required contract key."""
    rec = {
        "observation_id": "obs_000001",
        "camera_id": "cam_01",
        "timestamp": 142.37,
        # missing frame_index
        "object_id": "obj_cam01_000034",
        "object_type": "car",
        "confidence": 0.94,
        "bbox": [120.0, 80.0, 530.0, 310.0],
        "frame_path": "data/frames/cam_01/frame_004271.jpg",
        "crop_path": "data/crops/cam_01/obs_000001.jpg",
        "source_video": "data/videos/cam_01/vdo.avi",
    }
    is_valid, err = validate_observation_record(rec)
    assert is_valid is False
    assert "Missing required observation key: 'frame_index'" in err


def test_validate_observation_record_rejects_detection_id():
    """Verify schema validator strictly rejects detection_id (Stage 0 contract violation)."""
    rec = {
        "observation_id": "obs_000001",
        "camera_id": "cam_01",
        "timestamp": 142.37,
        "frame_index": 4271,
        "object_id": "obj_cam01_000034",
        "object_type": "car",
        "confidence": 0.94,
        "bbox": [120.0, 80.0, 530.0, 310.0],
        "frame_path": "data/frames/cam_01/frame_004271.jpg",
        "crop_path": "data/crops/cam_01/obs_000001.jpg",
        "source_video": "data/videos/cam_01/vdo.avi",
        "detection_id": "det_000001",
    }
    is_valid, err = validate_observation_record(rec)
    assert is_valid is False
    assert "detection_id is forbidden" in err


def test_validate_observation_record_rejects_extra_keys():
    """Verify schema validator rejects unapproved downstream fields."""
    rec = {
        "observation_id": "obs_000001",
        "camera_id": "cam_01",
        "timestamp": 142.37,
        "frame_index": 4271,
        "object_id": "obj_cam01_000034",
        "object_type": "car",
        "confidence": 0.94,
        "bbox": [120.0, 80.0, 530.0, 310.0],
        "frame_path": "data/frames/cam_01/frame_004271.jpg",
        "crop_path": "data/crops/cam_01/obs_000001.jpg",
        "source_video": "data/videos/cam_01/vdo.avi",
        "extra_field": "disallowed",
    }
    is_valid, err = validate_observation_record(rec)
    assert is_valid is False
    assert "Unexpected extra keys" in err


# ============================================================================
# 4. FULL DATASET ARTIFACT & REFERENTIAL INTEGRITY TESTS
# ============================================================================


def test_stage6_manifest_structure():
    """Verify data/observations/observation_manifest.json exists and adheres to contract."""
    manifest_p = Path("data/observations/observation_manifest.json")
    if not manifest_p.exists():
        pytest.skip("data/observations/observation_manifest.json does not exist")

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == "1.0"
    assert manifest["camera_count"] == 11
    assert manifest["input_tracked_records"] == 61039
    assert manifest["output_observation_records"] == 61039
    assert manifest["crop_count"] == 61039
    assert manifest["frames_referenced"] == 7287

    # Equality check
    assert manifest["input_tracked_records"] == manifest["output_observation_records"]
    assert manifest["output_observation_records"] == manifest["crop_count"]

    # Zero failures reported
    assert manifest["validation_results"]["failed_crops"] == 0
    assert manifest["validation_results"]["failed_observations"] == 0
    assert manifest["validation_results"]["verified_integrity"] is True

    # Check all 11 cameras present
    obs_per_cam = manifest["observations_per_camera"]
    crops_per_cam = manifest["crops_per_camera"]
    frames_per_cam = manifest["frames_referenced_per_camera"]
    assert len(obs_per_cam) == 11
    assert len(crops_per_cam) == 11

    expected_frames = {
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
    for cid, expected_n in expected_frames.items():
        assert frames_per_cam[cid] == expected_n
        assert obs_per_cam[cid] == crops_per_cam[cid]
        assert obs_per_cam[cid] > 0


def test_stage6_observations_jsonl_and_referential_integrity():
    """Verify data/observations/observations.jsonl satisfies all Stage 6 invariants."""
    obs_p = Path("data/observations/observations.jsonl")
    if not obs_p.exists():
        pytest.skip("data/observations/observations.jsonl does not exist")

    tracks_p = Path("data/tracks/tracks.jsonl")
    if not tracks_p.exists():
        pytest.skip("data/tracks/tracks.jsonl does not exist")

    # Load Stage 5 track records for 1:1 cross-validation
    track_records = []
    with open(tracks_p, "r", encoding="utf-8") as f:
        for line in f:
            track_records.append(json.loads(line))

    # Sort tracks deterministically to match observation assignment
    track_records.sort(key=lambda r: (r["camera_id"], r["frame_index"], r["object_id"]))

    obs_count = 0
    seen_obs_ids = set()
    camera_counts = {}

    with open(obs_p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obs_count += 1
            obs = json.loads(line)

            # 1. Observation ID sequence
            expected_obs_id = format_observation_id(obs_count)
            assert obs["observation_id"] == expected_obs_id
            assert obs["observation_id"] not in seen_obs_ids
            seen_obs_ids.add(obs["observation_id"])

            # 2. Referential integrity: 1:1 match with Stage 5 track record
            trk = track_records[idx]
            assert obs["camera_id"] == trk["camera_id"]
            assert obs["frame_index"] == trk["frame_index"]
            assert obs["object_id"] == trk["object_id"]
            assert obs["object_type"] == trk["object_type"]
            assert obs["confidence"] == trk["confidence"]
            assert obs["bbox"] == trk["bbox"]
            assert obs["frame_path"] == trk["frame_path"]
            assert obs["source_video"] == trk["source_video"]
            assert abs(obs["timestamp"] - trk["timestamp"]) < 1e-4

            # 3. Path existence check (sample every 1000th record)
            if idx % 1000 == 0:
                assert Path(obs["frame_path"]).exists(), f"Missing frame: {obs['frame_path']}"
                assert Path(obs["source_video"]).exists(), f"Missing video: {obs['source_video']}"
                crop_p = Path(obs["crop_path"])
                assert crop_p.exists(), f"Missing crop: {crop_p}"
                is_valid_crop, dims, _ = validate_crop_file(crop_p)
                assert is_valid_crop is True
                assert dims[0] > 0 and dims[1] > 0

            # 4. Strict scope check: No detection_id, no embeddings or FAISS vector fields
            assert set(obs.keys()) == set(FROZEN_OBSERVATION_KEYS), (
                f"Record {obs['observation_id']} keys {set(obs.keys())} do not match frozen contract {set(FROZEN_OBSERVATION_KEYS)}"
            )
            assert "detection_id" not in obs, f"Forbidden detection_id present in {obs['observation_id']}"
            for forbidden_field in ("embedding", "vector", "feature", "clip", "dino"):
                assert forbidden_field not in obs, f"Forbidden field '{forbidden_field}' in observation!"

            cid = obs["camera_id"]
            camera_counts[cid] = camera_counts.get(cid, 0) + 1

    assert obs_count == 61039
    assert len(seen_obs_ids) == 61039
    assert len(camera_counts) == 11

    # Verify total crop count on disk matches exactly 61,039
    crops_dir = Path("data/crops")
    total_crops_on_disk = sum(
        1 for cam_d in crops_dir.iterdir() if cam_d.is_dir()
        for f in cam_d.glob("*.jpg")
    )
    assert total_crops_on_disk == 61039
