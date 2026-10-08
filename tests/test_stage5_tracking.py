"""Tests for Stage 5: Multi-Object Tracking & Stable Per-Camera IDs."""

import json
import re
from pathlib import Path

import pytest

from backend.cv.tracking import (
    DEFAULT_MATCH_THRESH,
    DEFAULT_NEW_TRACK_THRESH,
    DEFAULT_TRACK_BUFFER,
    DEFAULT_TRACK_HIGH_THRESH,
    DEFAULT_TRACK_LOW_THRESH,
    OBJECT_ID_PATTERN,
    OfflineDetections,
    TrackingConfig,
    format_object_id,
    track_camera_detections,
    validate_track_record,
)


# ============================================================================
# 1. CONFIGURATION TESTS
# ============================================================================


def test_tracking_config_defaults():
    """Verify default tracking configuration values."""
    cfg = TrackingConfig()
    assert cfg.track_high_thresh == DEFAULT_TRACK_HIGH_THRESH
    assert cfg.track_low_thresh == DEFAULT_TRACK_LOW_THRESH
    assert cfg.new_track_thresh == DEFAULT_NEW_TRACK_THRESH
    assert cfg.track_buffer == DEFAULT_TRACK_BUFFER
    assert cfg.match_thresh == DEFAULT_MATCH_THRESH
    assert cfg.fuse_score is True


def test_tracking_config_custom_values():
    """Verify custom tracking configuration values."""
    cfg = TrackingConfig(
        track_high_thresh=0.5,
        track_low_thresh=0.2,
        new_track_thresh=0.4,
        track_buffer=15,
        match_thresh=0.7,
    )
    assert cfg.track_high_thresh == 0.5
    assert cfg.track_low_thresh == 0.2
    assert cfg.new_track_thresh == 0.4
    assert cfg.track_buffer == 15
    assert cfg.match_thresh == 0.7


@pytest.mark.parametrize("invalid_thresh", [0.0, -0.1, 1.5])
def test_tracking_config_rejects_invalid_high_thresh(invalid_thresh):
    """Verify out-of-range track_high_thresh is rejected."""
    with pytest.raises(ValueError, match="track_high_thresh must be in"):
        TrackingConfig(track_high_thresh=invalid_thresh)


@pytest.mark.parametrize("invalid_buffer", [0, -1, -10])
def test_tracking_config_rejects_invalid_buffer(invalid_buffer):
    """Verify non-positive track_buffer is rejected."""
    with pytest.raises(ValueError, match="track_buffer must be >= 1"):
        TrackingConfig(track_buffer=invalid_buffer)


# ============================================================================
# 2. OBJECT ID FORMAT & VALIDATION TESTS
# ============================================================================


def test_format_object_id():
    """Verify object ID formatting convention."""
    assert format_object_id("cam_01", 1) == "obj_cam01_000001"
    assert format_object_id("cam_01", 34) == "obj_cam01_000034"
    assert format_object_id("cam_11", 157) == "obj_cam11_000157"


@pytest.mark.parametrize(
    "valid_id",
    [
        "obj_cam01_000001",
        "obj_cam02_000034",
        "obj_cam11_000999",
    ],
)
def test_object_id_pattern_valid(valid_id):
    """Verify regex matches valid camera-local object IDs."""
    assert OBJECT_ID_PATTERN.match(valid_id) is not None


@pytest.mark.parametrize(
    "invalid_id",
    [
        "obj_000001",
        "cam01_000001",
        "obj_cam_01_000001",
        "obj_cam01_1",
        "track_01",
    ],
)
def test_object_id_pattern_invalid(invalid_id):
    """Verify regex rejects invalid object IDs."""
    assert OBJECT_ID_PATTERN.match(invalid_id) is None


def test_validate_track_record_valid():
    """Verify validator accepts well-formed track record."""
    rec = {
        "camera_id": "cam_01",
        "frame_index": 30,
        "timestamp": 3.0,
        "object_id": "obj_cam01_000005",
        "object_type": "car",
        "class_id": 2,
        "confidence": 0.88,
        "bbox": [100.0, 200.0, 300.0, 400.0],
        "frame_path": "data/frames/cam_01/frame_000030.jpg",
        "source_video": "data/videos/cam_01/vdo.avi",
        "detection_id": "det_000010",
    }
    is_valid, err = validate_track_record(rec)
    assert is_valid is True
    assert err is None


def test_validate_track_record_missing_key():
    """Verify validator rejects track record with missing required key."""
    rec = {
        "camera_id": "cam_01",
        "frame_index": 30,
        "timestamp": 3.0,
        # missing object_id
        "object_type": "car",
        "class_id": 2,
        "confidence": 0.88,
        "bbox": [100.0, 200.0, 300.0, 400.0],
        "frame_path": "data/frames/cam_01/frame_000030.jpg",
        "source_video": "data/videos/cam_01/vdo.avi",
    }
    is_valid, err = validate_track_record(rec)
    assert is_valid is False
    assert "Missing required key: 'object_id'" in err


def test_validate_track_record_camera_mismatch():
    """Verify validator detects mismatch between camera_id and object_id."""
    rec = {
        "camera_id": "cam_01",
        "frame_index": 30,
        "timestamp": 3.0,
        "object_id": "obj_cam02_000005",  # cam02 in cam_01 record
        "object_type": "car",
        "class_id": 2,
        "confidence": 0.88,
        "bbox": [100.0, 200.0, 300.0, 400.0],
        "frame_path": "data/frames/cam_01/frame_000030.jpg",
        "source_video": "data/videos/cam_01/vdo.avi",
    }
    is_valid, err = validate_track_record(rec)
    assert is_valid is False
    assert "does not match camera_id" in err


# ============================================================================
# 3. DETERMINISM & OFFLINE ADAPTER TESTS
# ============================================================================


def test_offline_detections_adapter():
    """Verify OfflineDetections properties and slicing."""
    dets = OfflineDetections(
        xyxy=[[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
        conf=[0.9, 0.8],
        cls=[2, 0],
    )
    assert len(dets) == 2
    assert dets.xywh.shape == (2, 4)
    # Check center of first box
    assert round(dets.xywh[0, 0], 1) == 20.0
    assert round(dets.xywh[0, 1], 1) == 30.0
    assert round(dets.xywh[0, 2], 1) == 20.0
    assert round(dets.xywh[0, 3], 1) == 20.0

    sliced = dets[[True, False]]
    assert len(sliced) == 1
    assert sliced.conf[0] == 0.9


def test_tracking_determinism_on_synthetic_sequence():
    """Verify running tracking twice on identical input produces identical tracks."""
    config = TrackingConfig()
    camera_id = "cam_01"
    frames = {
        0: [
            {"camera_id": "cam_01", "frame_index": 0, "timestamp": 0.0, "bbox": [100.0, 100.0, 200.0, 200.0], "confidence": 0.9, "class_id": 2, "class_name": "car", "frame_path": "f0.jpg", "source_video": "v.avi", "detection_id": "d1"},
            {"camera_id": "cam_01", "frame_index": 0, "timestamp": 0.0, "bbox": [400.0, 400.0, 500.0, 500.0], "confidence": 0.85, "class_id": 2, "class_name": "car", "frame_path": "f0.jpg", "source_video": "v.avi", "detection_id": "d2"},
        ],
        3: [
            {"camera_id": "cam_01", "frame_index": 3, "timestamp": 0.3, "bbox": [102.0, 101.0, 202.0, 201.0], "confidence": 0.92, "class_id": 2, "class_name": "car", "frame_path": "f3.jpg", "source_video": "v.avi", "detection_id": "d3"},
            {"camera_id": "cam_01", "frame_index": 3, "timestamp": 0.3, "bbox": [401.0, 402.0, 501.0, 502.0], "confidence": 0.88, "class_id": 2, "class_name": "car", "frame_path": "f3.jpg", "source_video": "v.avi", "detection_id": "d4"},
        ],
    }

    recs1, summ1 = track_camera_detections(camera_id, frames, config)
    recs2, summ2 = track_camera_detections(camera_id, frames, config)

    assert len(recs1) == len(recs2)
    assert summ1 == summ2
    for r1, r2 in zip(recs1, recs2):
        assert r1 == r2


# ============================================================================
# 4. FULL DATASET ARTIFACT & INVARIANT TESTS
# ============================================================================


def test_stage5_track_manifest_structure():
    """Verify data/tracks/track_manifest.json exists and adheres to contract."""
    manifest_path = Path("data/tracks/track_manifest.json")
    if not manifest_path.exists():
        pytest.skip("data/tracks/track_manifest.json does not exist")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == "1.0"
    assert manifest["tracker_name"] == "ByteTrack"
    assert manifest["input_detection_count"] == 68906
    assert manifest["camera_count"] == 11
    assert manifest["frame_count"] == 7287
    assert manifest["tracked_detection_count"] == 61039
    assert manifest["discarded_detection_count"] == 7867
    assert manifest["unique_object_count"] == 933

    # Verify camera coverage across all 11 cameras
    objs_per_cam = manifest["objects_per_camera"]
    tracked_per_cam = manifest["tracked_per_camera"]
    discarded_per_cam = manifest["discarded_per_camera"]
    detections_per_cam = manifest["detections_per_camera"]
    frames_per_cam = manifest["frames_per_camera"]

    assert len(objs_per_cam) == 11
    expected_cams = [f"cam_{i:02d}" for i in range(1, 12)]
    for cid in expected_cams:
        assert cid in objs_per_cam
        assert objs_per_cam[cid] > 0
        assert tracked_per_cam[cid] + discarded_per_cam[cid] == detections_per_cam[cid]

    # Verify sum across all cameras
    assert sum(tracked_per_cam.values()) == 61039
    assert sum(discarded_per_cam.values()) == 7867
    assert sum(detections_per_cam.values()) == 68906
    assert sum(frames_per_cam.values()) == 7287


def test_stage5_tracks_jsonl_invariants():
    """Verify data/tracks/tracks.jsonl satisfies all tracking invariants."""
    tracks_path = Path("data/tracks/tracks.jsonl")
    if not tracks_path.exists():
        pytest.skip("data/tracks/tracks.jsonl does not exist")

    # Authoritative frames map
    frame_index_path = Path("data/frames/frame_index.jsonl")
    frame_map = {}
    if frame_index_path.exists():
        with open(frame_index_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                frame_map[(rec["camera_id"], rec["frame_index"])] = rec

    count = 0
    seen_cameras = set()
    camera_objects = {f"cam_{i:02d}": set() for i in range(1, 12)}
    obj_to_camera = {}
    object_frame_sequences = {}  # object_id -> list of frame_indices
    frame_object_pairs = set()

    with open(tracks_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            count += 1
            rec = json.loads(line)
            cid = rec["camera_id"]
            seen_cameras.add(cid)
            oid = rec["object_id"]
            fidx = rec["frame_index"]

            # 1. Camera Isolation: No cross-camera object IDs
            if oid not in obj_to_camera:
                obj_to_camera[oid] = cid
            else:
                assert obj_to_camera[oid] == cid, f"Cross-camera ID detected: {oid} in {cid} and {obj_to_camera[oid]}"
            camera_objects[cid].add(oid)

            # 2. Frame uniqueness: At most one detection per object per frame
            pair = (cid, fidx, oid)
            assert pair not in frame_object_pairs, f"Duplicate object assignment in frame: {pair}"
            frame_object_pairs.add(pair)

            # 3. Frame ordering per object: strictly increasing
            if oid not in object_frame_sequences:
                object_frame_sequences[oid] = []
            if object_frame_sequences[oid]:
                prev_fidx = object_frame_sequences[oid][-1]
                assert fidx > prev_fidx, f"Non-monotonic frame sequence for {oid}: {prev_fidx} -> {fidx}"
            object_frame_sequences[oid].append(fidx)

            # 4. Bounding Box validity
            bbox = rec["bbox"]
            assert len(bbox) == 4
            assert bbox[0] < bbox[2]
            assert bbox[1] < bbox[3]
            assert bbox[0] >= 0.0
            assert bbox[1] >= 0.0

            # 5. Timestamp consistency
            if (cid, fidx) in frame_map and idx % 1000 == 0:
                expected_ts = frame_map[(cid, fidx)]["timestamp"]
                assert abs(rec["timestamp"] - expected_ts) < 1e-4

    assert count == 61039
    assert len(seen_cameras) == 11
    assert len(obj_to_camera) == 933

    # Check that each camera's object numbering starts at 000001
    for cid in sorted(camera_objects.keys()):
        objs = sorted(camera_objects[cid])
        clean_cam = cid.replace("_", "")
        expected_first = f"obj_{clean_cam}_000001"
        assert objs[0] == expected_first, f"{cid} first object is {objs[0]}, expected {expected_first}"
        assert len(objs) == len(camera_objects[cid])
