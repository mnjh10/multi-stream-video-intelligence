"""Focused unit tests for Stage 1: Video Ingestion & Camera Inventory."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.cv.discovery import (
    assign_camera_ids,
    discover_video_sources,
    normalize_repo_path,
)
from backend.cv.inventory import build_camera_inventory, save_camera_inventory
from backend.cv.metadata import extract_video_metadata
from backend.cv.validator import validate_video_metadata


def test_empty_input_directory(tmp_path: Path):
    """Test behavior when input directory is empty or does not exist."""
    empty_dir = tmp_path / "videos"
    empty_dir.mkdir()

    discovered = discover_video_sources(video_dir=empty_dir, repo_root=tmp_path)
    assert discovered == []

    inventory = build_camera_inventory(video_dir=empty_dir, repo_root=tmp_path)
    assert inventory["total_cameras"] == 0
    assert inventory["valid_cameras"] == 0
    assert inventory["invalid_cameras"] == 0
    assert inventory["cameras"] == []
    assert inventory["schema_version"] == "1.0"
    assert "generated_at" not in inventory


def test_nonexistent_directory(tmp_path: Path):
    """Test behavior when input directory does not exist."""
    nonexistent = tmp_path / "does_not_exist"
    discovered = discover_video_sources(video_dir=nonexistent, repo_root=tmp_path)
    assert discovered == []


def test_deterministic_discovery_ordering(tmp_path: Path):
    """Test that video discovery ordering is strictly deterministic."""
    video_dir = tmp_path / "data" / "videos"
    video_dir.mkdir(parents=True)

    # Create dummy files in non-alphabetical creation order
    # (Note: plain text files with .mp4 suffix, used solely for path discovery testing)
    (video_dir / "cam_03").mkdir()
    (video_dir / "cam_03" / "video.mp4").write_text("dummy")

    (video_dir / "cam_01").mkdir()
    (video_dir / "cam_01" / "video.mp4").write_text("dummy")

    (video_dir / "cam_02").mkdir()
    (video_dir / "cam_02" / "video.mp4").write_text("dummy")

    (video_dir / "zebra_cam.mp4").write_text("dummy")
    (video_dir / "alpha_cam.mp4").write_text("dummy")

    # Run discovery multiple times and verify order
    discovered_1 = discover_video_sources(video_dir=video_dir, repo_root=tmp_path)
    discovered_2 = discover_video_sources(video_dir=video_dir, repo_root=tmp_path)

    assert len(discovered_1) == 5
    rel_paths_1 = [d["relative_path"] for d in discovered_1]
    rel_paths_2 = [d["relative_path"] for d in discovered_2]

    # Verify identical ordering across runs
    assert rel_paths_1 == rel_paths_2

    # Verify strictly sorted lexicographically
    assert rel_paths_1 == sorted(rel_paths_1, key=lambda p: (p.lower(), p))


def test_camera_id_assignment_canonical():
    """Test normalization of canonical camera names."""
    dummy_sources = [
        {"source_name": "cam_01", "relative_path": "data/videos/cam_01/video.mp4"},
        {"source_name": "cam02", "relative_path": "data/videos/cam02/video.mp4"},
        {"source_name": "CAM_3", "relative_path": "data/videos/CAM_3/video.mp4"},
        {"source_name": "cam-04", "relative_path": "data/videos/cam-04/video.mp4"},
    ]
    assigned = assign_camera_ids(dummy_sources)
    ids = [a["camera_id"] for a in assigned]
    assert ids == ["cam_01", "cam_02", "cam_03", "cam_04"]


def test_camera_id_assignment_arbitrary_names():
    """Test sequential canonical ID generation for arbitrary/unmatched names."""
    dummy_sources = [
        {"source_name": "entrance", "relative_path": "data/videos/entrance/feed.mp4"},
        {"source_name": "gate_north", "relative_path": "data/videos/gate_north/feed.mp4"},
        {"source_name": "parking", "relative_path": "data/videos/parking/feed.mp4"},
    ]
    assigned = assign_camera_ids(dummy_sources)
    ids = [a["camera_id"] for a in assigned]
    assert ids == ["cam_01", "cam_02", "cam_03"]


def test_camera_id_collision_avoidance():
    """Test that explicit IDs and auto-generated IDs do not collide."""
    dummy_sources = [
        {"source_name": "cam_01", "relative_path": "data/videos/cam_01/video.mp4"},
        {"source_name": "lobby", "relative_path": "data/videos/lobby/video.mp4"},
        {"source_name": "cam_02", "relative_path": "data/videos/cam_02/video.mp4"},
    ]
    assigned = assign_camera_ids(dummy_sources)
    ids = [a["camera_id"] for a in assigned]
    # cam_01 and cam_02 are taken, lobby should receive next available sequential ID (cam_03)
    assert set(ids) == {"cam_01", "cam_02", "cam_03"}
    assert len(ids) == len(set(ids))


def test_single_video_in_camera_directory_valid(tmp_path: Path):
    """Test that exactly one video in a camera directory is processed normally and valid."""
    cam_dir = tmp_path / "data" / "videos" / "cam_01"
    cam_dir.mkdir(parents=True)
    video_file = cam_dir / "video.mp4"
    video_file.write_bytes(b"mock_video_bytes")

    # Mock OpenCV VideoCapture to simulate a valid video stream without creating synthetic video files
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # 720p frame
    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, mock_frame)

    def mock_get(prop_id):
        import cv2

        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 1280.0
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 720.0
        elif prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 1800.0
        elif prop_id == cv2.CAP_PROP_FOURCC:
            return 0.0
        return 0.0

    mock_cap.get.side_effect = mock_get

    with patch("cv2.VideoCapture", return_value=mock_cap):
        inventory = build_camera_inventory(
            video_dir=tmp_path / "data" / "videos",
            repo_root=tmp_path,
        )

    assert inventory["total_cameras"] == 1
    assert inventory["valid_cameras"] == 1
    assert inventory["invalid_cameras"] == 0
    cam = inventory["cameras"][0]
    assert cam["camera_id"] == "cam_01"
    assert cam["validation_status"] == "valid"
    assert cam["width"] == 1280
    assert cam["height"] == 720
    assert cam["fps"] == 30.0
    assert cam["frame_count"] == 1800
    assert cam["duration_seconds"] == 60.0
    assert cam["source_video"] == "data/videos/cam_01/video.mp4"
    assert cam["candidate_files"] == ["data/videos/cam_01/video.mp4"]


def test_multiple_videos_in_camera_directory_ambiguity_error(tmp_path: Path):
    """Test that multiple videos in the same camera directory trigger ambiguity error without silent discard."""
    cam_dir = tmp_path / "data" / "videos" / "cam_01"
    cam_dir.mkdir(parents=True)
    morning = cam_dir / "morning.mp4"
    morning.write_bytes(b"morning_data")
    afternoon = cam_dir / "afternoon.mp4"
    afternoon.write_bytes(b"afternoon_data")

    inventory = build_camera_inventory(
        video_dir=tmp_path / "data" / "videos",
        repo_root=tmp_path,
    )

    assert inventory["total_cameras"] == 1
    assert inventory["valid_cameras"] == 0
    assert inventory["invalid_cameras"] == 1
    cam = inventory["cameras"][0]
    assert cam["camera_id"] == "cam_01"
    assert cam["validation_status"] == "invalid"
    assert "Multiple video files found in camera directory" in cam["error_message"]
    assert "Multi-video ambiguity is not permitted" in cam["error_message"]
    # Verify no file was silently discarded
    expected_candidates = [
        "data/videos/cam_01/afternoon.mp4",
        "data/videos/cam_01/morning.mp4",
    ]
    assert cam["candidate_files"] == expected_candidates


def test_manifest_is_strictly_deterministic(tmp_path: Path):
    """Test that running manifest generation twice on the same inputs produces identical JSON content."""
    cam_dir = tmp_path / "data" / "videos" / "cam_01"
    cam_dir.mkdir(parents=True)
    video_file = cam_dir / "video.mp4"
    video_file.write_bytes(b"mock_content")

    # Run 1
    inventory_1 = build_camera_inventory(video_dir=tmp_path / "data" / "videos", repo_root=tmp_path)
    out_1 = tmp_path / "out_1.json"
    save_camera_inventory(inventory_1, output_path=out_1, repo_root=tmp_path)

    # Run 2
    inventory_2 = build_camera_inventory(video_dir=tmp_path / "data" / "videos", repo_root=tmp_path)
    out_2 = tmp_path / "out_2.json"
    save_camera_inventory(inventory_2, output_path=out_2, repo_root=tmp_path)

    content_1 = out_1.read_text(encoding="utf-8")
    content_2 = out_2.read_text(encoding="utf-8")

    # Byte-for-byte identical output
    assert content_1 == content_2
    assert "generated_at" not in content_1


def test_malformed_empty_file_handling(tmp_path: Path):
    """Test validation and error handling for 0-byte video files."""
    empty_file = tmp_path / "empty_video.mp4"
    empty_file.touch()

    meta = extract_video_metadata(video_path=empty_file, camera_id="cam_01", repo_root=tmp_path)
    assert meta["camera_id"] == "cam_01"
    assert meta["file_size_bytes"] == 0
    assert meta["readable"] is False
    assert meta["validation_status"] == "invalid"
    assert "empty" in meta["error_message"].lower()


def test_malformed_corrupt_file_handling(tmp_path: Path):
    """Test validation and error handling for non-video text files renamed as .mp4."""
    corrupt_file = tmp_path / "corrupt_video.mp4"
    corrupt_file.write_text("This is plain text and not a video stream.")

    meta = extract_video_metadata(video_path=corrupt_file, camera_id="cam_01", repo_root=tmp_path)
    assert meta["camera_id"] == "cam_01"
    assert meta["file_size_bytes"] > 0
    assert meta["readable"] is False
    assert meta["validation_status"] == "invalid"
    assert meta["error_message"] is not None


def test_missing_file_handling(tmp_path: Path):
    """Test extraction on non-existent file."""
    missing_file = tmp_path / "nonexistent.mp4"
    meta = extract_video_metadata(video_path=missing_file, camera_id="cam_01", repo_root=tmp_path)
    assert meta["camera_id"] == "cam_01"
    assert meta["validation_status"] == "invalid"
    assert "not found" in meta["error_message"].lower()


def test_manifest_serialization(tmp_path: Path):
    """Test building and writing manifest JSON to disk."""
    inventory = build_camera_inventory(video_dir=tmp_path / "videos", repo_root=tmp_path)
    manifest_path = tmp_path / "observations" / "camera_inventory.json"
    saved = save_camera_inventory(inventory, output_path=manifest_path, repo_root=tmp_path)

    assert saved.exists()
    with open(saved, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["schema_version"] == "1.0"
    assert "generated_at" not in data
    assert "video_directory" in data
    assert data["total_cameras"] == 0
    assert data["valid_cameras"] == 0
    assert data["invalid_cameras"] == 0
    assert data["cameras"] == []


def test_repo_relative_path_normalization(tmp_path: Path):
    """Test that paths are converted to POSIX repository-relative paths without Windows drive letters."""
    nested_path = tmp_path / "data" / "videos" / "cam_01" / "video.mp4"
    rel_path = normalize_repo_path(nested_path, repo_root=tmp_path)

    assert rel_path == "data/videos/cam_01/video.mp4"
    assert "\\" not in rel_path
    assert ":" not in rel_path
