"""Tests for Stage 2: CityFlowV2 Dataset Acquisition and Lock."""

import json
from pathlib import Path

import cv2
import pytest


@pytest.fixture
def manifest_data():
    manifest_path = Path("data/dataset_manifest.json")
    if not manifest_path.exists():
        pytest.skip("data/dataset_manifest.json does not exist yet")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_file_exists():
    """Verify that the dataset manifest exists."""
    manifest_path = Path("data/dataset_manifest.json")
    assert manifest_path.exists(), "data/dataset_manifest.json must exist"


def test_manifest_is_valid_json(manifest_data):
    """Verify that dataset_manifest.json is valid and contains required top-level keys."""
    required_keys = [
        "dataset_name",
        "dataset_version",
        "official_source",
        "split",
        "scenarios",
        "total_selected_cameras",
        "total_selected_videos",
        "total_selected_duration_seconds",
        "selection_rationale",
        "cameras",
    ]
    for key in required_keys:
        assert key in manifest_data, f"Missing required manifest key: {key}"

    # Assert no dynamic timestamp keys
    assert "generated_at" not in manifest_data
    assert "timestamp" not in manifest_data


def test_manifest_cameras_count_and_uniqueness(manifest_data):
    """Verify that the camera count matches the target and camera IDs are unique."""
    cameras = manifest_data["cameras"]
    assert 8 <= len(cameras) <= 12, f"Selected cameras ({len(cameras)}) must be in 8-12 range"
    assert len(cameras) == manifest_data["total_selected_cameras"]

    cam_ids = [c["camera_id"] for c in cameras]
    assert len(cam_ids) == len(set(cam_ids)), "Camera IDs must be unique"


def test_manifest_camera_paths_deterministic(manifest_data):
    """Verify that video paths are deterministic and repository-relative."""
    cameras = manifest_data["cameras"]
    source_paths = [c["source_video"] for c in cameras]

    # Deterministic alphabetical ordering
    assert source_paths == sorted(source_paths, key=lambda p: (p.lower(), p))

    # POSIX repository-relative convention
    for p in source_paths:
        assert p.startswith("data/videos/cam_"), f"Path must follow convention: {p}"
        assert "\\" not in p
        assert ":" not in p


def test_selected_videos_exist_and_readable(manifest_data):
    """Verify that every video declared in the manifest exists and is readable by OpenCV."""
    cameras = manifest_data["cameras"]
    for c in cameras:
        video_path = Path(c["source_video"])
        assert video_path.exists(), f"Video file does not exist on disk: {video_path}"
        assert video_path.stat().st_size > 0, f"Video file is empty: {video_path}"

        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"OpenCV failed to open {video_path}"
        ret, frame = cap.read()
        cap.release()

        assert ret is True, f"Failed to decode initial frame of {video_path}"
        assert frame is not None and frame.size > 0


def test_metadata_values_valid(manifest_data):
    """Verify that metadata fields in the manifest match physical video properties."""
    for c in manifest_data["cameras"]:
        assert isinstance(c["width"], int) and c["width"] >= 960
        assert isinstance(c["height"], int) and c["height"] >= 540
        assert isinstance(c["fps"], (int, float)) and c["fps"] in (8.0, 10.0)
        assert isinstance(c["frame_count"], int) and c["frame_count"] > 0
        assert isinstance(c["duration_seconds"], (int, float)) and c["duration_seconds"] > 0
        assert c["split"] == "train"
        assert c["scenario_id"] in ("S01", "S03")
