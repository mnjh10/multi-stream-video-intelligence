"""Tests for Stage 3: Deterministic Frame Sampling & Timestamp Indexing."""

import json
from pathlib import Path

import cv2
import pytest

from backend.cv.sampling import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_SAMPLE_FPS,
    SamplingConfig,
    compute_sample_indices,
    compute_timestamp,
    sample_camera_video,
)


# ============================================================================
# 1. SAMPLING CONFIGURATION & VALIDATION TESTS
# ============================================================================


def test_sampling_config_valid_defaults():
    """Verify default sampling configuration is valid."""
    config = SamplingConfig()
    assert config.sample_fps == DEFAULT_SAMPLE_FPS
    assert config.jpeg_quality == DEFAULT_JPEG_QUALITY
    assert config.output_dir == "data/frames"
    assert config.index_file == "data/frames/frame_index.jsonl"
    assert config.manifest_file == "data/frames/sampling_manifest.json"


def test_sampling_config_custom_values():
    """Verify custom sampling configuration is accepted."""
    config = SamplingConfig(sample_fps=5.0, jpeg_quality=90)
    assert config.sample_fps == 5.0
    assert config.jpeg_quality == 90


@pytest.mark.parametrize("invalid_fps", [0.0, -1.0, -5.5])
def test_sampling_config_rejects_non_positive_fps(invalid_fps):
    """Verify that zero or negative sample_fps raises ValueError."""
    with pytest.raises(ValueError, match="sample_fps must be positive"):
        SamplingConfig(sample_fps=invalid_fps)


@pytest.mark.parametrize("invalid_quality", [0, -1, 101, 150])
def test_sampling_config_rejects_invalid_jpeg_quality(invalid_quality):
    """Verify that out-of-range jpeg_quality raises ValueError."""
    with pytest.raises(ValueError, match="jpeg_quality must be in"):
        SamplingConfig(jpeg_quality=invalid_quality)


# ============================================================================
# 2. SAMPLING ALGORITHM & INDEX COMPUTATION TESTS
# ============================================================================


def test_compute_sample_indices_10fps_3sample():
    """Verify deterministic frame sampling schedule for 10 FPS source at 3 FPS target."""
    # 30 frames = 3.0 seconds. Desired samples: 9 frames total.
    indices = compute_sample_indices(frame_count=30, source_fps=10.0, sample_fps=3.0)
    # k=0: round(0) = 0
    # k=1: round(1 * 10/3) = 3
    # k=2: round(2 * 10/3) = 7
    # k=3: round(3 * 10/3) = 10
    # k=4: round(4 * 10/3) = 13
    # k=5: round(5 * 10/3) = 17
    # k=6: round(6 * 10/3) = 20
    # k=7: round(7 * 10/3) = 23
    # k=8: round(8 * 10/3) = 27
    expected = [0, 3, 7, 10, 13, 17, 20, 23, 27]
    assert indices == expected
    assert len(indices) == 9
    assert indices == sorted(list(set(indices))), "Indices must be strictly increasing and unique"


def test_compute_sample_indices_8fps_3sample():
    """Verify deterministic frame sampling schedule for 8 FPS source (cam_11) at 3 FPS target."""
    # 24 frames = 3.0 seconds. Desired samples: 9 frames total.
    indices = compute_sample_indices(frame_count=24, source_fps=8.0, sample_fps=3.0)
    # k=0: round(0) = 0
    # k=1: round(1 * 8/3) = 3
    # k=2: round(2 * 8/3) = 5
    # k=3: round(3 * 8/3) = 8
    # k=4: round(4 * 8/3) = 11
    # k=5: round(5 * 8/3) = 13
    # k=6: round(6 * 8/3) = 16
    # k=7: round(7 * 8/3) = 19
    # k=8: round(8 * 8/3) = 21
    expected = [0, 3, 5, 8, 11, 13, 16, 19, 21]
    assert indices == expected
    assert len(indices) == 9
    assert indices == sorted(list(set(indices))), "Indices must be strictly increasing and unique"


def test_frame_0_always_included_for_any_positive_count():
    """Verify that frame_index == 0 is always included as the first element for any frame_count > 0."""
    for fc in [1, 5, 10, 100, 2110]:
        for src_fps in [8.0, 10.0, 25.0, 30.0]:
            for smp_fps in [1.0, 3.0, 5.0, 8.0, 10.0, 30.0]:
                indices = compute_sample_indices(frame_count=fc, source_fps=src_fps, sample_fps=smp_fps)
                assert len(indices) > 0
                assert indices[0] == 0, f"First frame must be 0 for fc={fc}, src_fps={src_fps}, smp_fps={smp_fps}"


def test_final_sampled_index_never_exceeds_n_minus_one():
    """Verify that the maximum sampled frame_index never exceeds frame_count - 1."""
    test_cases = [
        (1, 10.0, 3.0),
        (2, 10.0, 3.0),
        (3, 10.0, 3.0),
        (4, 10.0, 3.0),
        (10, 10.0, 3.0),
        (23798, 10.0, 3.0),
        (1928, 8.0, 3.0),
        (100, 8.0, 7.5),
        (50, 10.0, 15.0),
    ]
    for fc, src_fps, smp_fps in test_cases:
        indices = compute_sample_indices(frame_count=fc, source_fps=src_fps, sample_fps=smp_fps)
        assert max(indices) <= fc - 1, f"Max index {max(indices)} exceeds {fc - 1}"


def test_compute_sample_indices_sample_rate_gte_source_rate():
    """Verify that when sample_fps >= source_fps, all frames are selected without duplicates."""
    # Equal to source FPS
    indices_equal = compute_sample_indices(frame_count=10, source_fps=10.0, sample_fps=10.0)
    assert indices_equal == list(range(10))

    # Greater than source FPS
    indices_greater = compute_sample_indices(frame_count=10, source_fps=10.0, sample_fps=20.0)
    assert indices_greater == list(range(10))


def test_compute_sample_indices_zero_or_empty():
    """Verify edge case of 0 frames returns empty list."""
    assert compute_sample_indices(frame_count=0, source_fps=10.0, sample_fps=3.0) == []


def test_compute_sample_indices_invalid_inputs():
    """Verify ValueError on non-positive source_fps or sample_fps."""
    with pytest.raises(ValueError):
        compute_sample_indices(frame_count=100, source_fps=-10.0, sample_fps=3.0)
    with pytest.raises(ValueError):
        compute_sample_indices(frame_count=100, source_fps=10.0, sample_fps=0.0)


# ============================================================================
# 3. TIMESTAMP COMPUTATION & SEMANTICS TESTS
# ============================================================================


def test_timestamp_formula_10fps():
    """Verify timestamp == frame_index / source_fps for 10 FPS source."""
    source_fps = 10.0
    for frame_idx in [0, 3, 7, 10, 15, 30, 100]:
        ts = compute_timestamp(frame_index=frame_idx, source_fps=source_fps)
        expected = round(frame_idx / source_fps, 4)
        assert ts == expected


def test_timestamp_formula_8fps():
    """Verify timestamp == frame_index / source_fps for 8 FPS source (cam_11)."""
    source_fps = 8.0
    # frame 0 -> 0.0, frame 3 -> 0.375, frame 5 -> 0.625, frame 8 -> 1.0
    assert compute_timestamp(0, source_fps) == 0.0
    assert compute_timestamp(3, source_fps) == 0.375
    assert compute_timestamp(5, source_fps) == 0.625
    assert compute_timestamp(8, source_fps) == 1.0
    assert compute_timestamp(11, source_fps) == 1.375


def test_timestamp_unrounded_vs_rounded():
    """Verify compute_timestamp precision behavior (rounded vs raw unrounded)."""
    # 1/10 raw has binary float precision residue
    raw = compute_timestamp(frame_index=3, source_fps=10.0, precision=None)
    rounded = compute_timestamp(frame_index=3, source_fps=10.0, precision=4)
    assert raw == 3 / 10.0
    assert rounded == 0.3
    # 8th is power of 2: exactly represented
    raw8 = compute_timestamp(frame_index=3, source_fps=8.0, precision=None)
    rounded8 = compute_timestamp(frame_index=3, source_fps=8.0, precision=4)
    assert raw8 == 0.375
    assert rounded8 == 0.375



def test_timestamp_invalid_inputs():
    """Verify compute_timestamp rejects negative frame_index or non-positive fps."""
    with pytest.raises(ValueError):
        compute_timestamp(frame_index=-1, source_fps=10.0)
    with pytest.raises(ValueError):
        compute_timestamp(frame_index=10, source_fps=0.0)


# ============================================================================
# 4. FPS HETEROGENEITY TESTS (CAM_01 vs CAM_11)
# ============================================================================


def test_fps_heterogeneity_distinct_timelines():
    """Verify that cam_01 (10 FPS) and cam_11 (8 FPS) yield distinct timestamps for frame 3."""
    ts_cam01 = compute_timestamp(frame_index=3, source_fps=10.0)
    ts_cam11 = compute_timestamp(frame_index=3, source_fps=8.0)
    assert ts_cam01 == 0.3
    assert ts_cam11 == 0.375
    assert ts_cam01 != ts_cam11, "Implementation must not assume 10 FPS globally"


# ============================================================================
# 5. INTEGRATION SAMPLING & FRAME EXTRACTION TESTS
# ============================================================================


@pytest.fixture
def test_output_dir(tmp_path):
    """Temporary directory for frame output in isolated unit tests."""
    return tmp_path / "test_frames"


def test_sample_camera_video_with_real_video(test_output_dir):
    """Verify sample_camera_video extracts valid, unresized frames from a real video."""
    video_path = Path("data/videos/cam_05/vdo.avi")
    if not video_path.exists():
        pytest.skip("data/videos/cam_05/vdo.avi does not exist")

    # Use a small test slice: 30 frames
    cam_record = {
        "camera_id": "cam_05",
        "source_video": "data/videos/cam_05/vdo.avi",
        "fps": 10.0,
        "frame_count": 30,  # test only first 30 frames
        "width": 1280,
        "height": 960,
    }
    config = SamplingConfig(
        sample_fps=3.0,
        output_dir=str(test_output_dir),
    )

    res = sample_camera_video(camera_record=cam_record, config=config)

    assert res["camera_id"] == "cam_05"
    assert res["sampled_frames_count"] == 9
    assert len(res["frame_records"]) == 9

    # Verify frame index and timestamp correspondence
    expected_indices = [0, 3, 7, 10, 13, 17, 20, 23, 27]
    for r, exp_idx in zip(res["frame_records"], expected_indices):
        assert r["frame_index"] == exp_idx
        assert r["timestamp"] == round(exp_idx / 10.0, 4)
        assert r["frame_path"].endswith(f"frame_{exp_idx:06d}.jpg")

        # Verify disk file exists and dimensions match native source resolution
        frame_file = test_output_dir / "cam_05" / f"frame_{exp_idx:06d}.jpg"
        assert frame_file.exists()
        assert frame_file.stat().st_size > 0

        img = cv2.imread(str(frame_file))
        assert img is not None
        h, w = img.shape[:2]
        assert w == 1280, "Frame must preserve native width (no resizing)"
        assert h == 960, "Frame must preserve native height (no resizing)"


def test_sample_cam11_8fps_native_dimensions(test_output_dir):
    """Verify cam_11 (8 FPS, 1920x1080) samples using 8 FPS timeline without assuming 10 FPS."""
    video_path = Path("data/videos/cam_11/vdo.avi")
    if not video_path.exists():
        pytest.skip("data/videos/cam_11/vdo.avi does not exist")

    cam_record = {
        "camera_id": "cam_11",
        "source_video": "data/videos/cam_11/vdo.avi",
        "fps": 8.0,
        "frame_count": 24,  # test first 24 frames (3 seconds)
        "width": 1920,
        "height": 1080,
    }
    config = SamplingConfig(
        sample_fps=3.0,
        output_dir=str(test_output_dir),
    )

    res = sample_camera_video(camera_record=cam_record, config=config)

    assert res["camera_id"] == "cam_11"
    assert res["source_fps"] == 8.0
    assert res["sampled_frames_count"] == 9

    expected_indices = [0, 3, 5, 8, 11, 13, 16, 19, 21]
    for r, exp_idx in zip(res["frame_records"], expected_indices):
        assert r["frame_index"] == exp_idx
        assert r["timestamp"] == round(exp_idx / 8.0, 4)

        frame_file = test_output_dir / "cam_11" / f"frame_{exp_idx:06d}.jpg"
        assert frame_file.exists()
        img = cv2.imread(str(frame_file))
        assert img is not None
        h, w = img.shape[:2]
        assert w == 1920, "Frame must preserve native width (1920)"
        assert h == 1080, "Frame must preserve native height (1080)"


def test_sampling_determinism(test_output_dir):
    """Verify running sampling twice produces identical frame records and metadata."""
    video_path = Path("data/videos/cam_05/vdo.avi")
    if not video_path.exists():
        pytest.skip("data/videos/cam_05/vdo.avi does not exist")

    cam_record = {
        "camera_id": "cam_05",
        "source_video": "data/videos/cam_05/vdo.avi",
        "fps": 10.0,
        "frame_count": 20,
        "width": 1280,
        "height": 960,
    }
    config1 = SamplingConfig(sample_fps=3.0, output_dir=str(test_output_dir / "run1"))
    config2 = SamplingConfig(sample_fps=3.0, output_dir=str(test_output_dir / "run2"))

    res1 = sample_camera_video(camera_record=cam_record, config=config1)
    res2 = sample_camera_video(camera_record=cam_record, config=config2)

    assert len(res1["frame_records"]) == len(res2["frame_records"])
    for r1, r2 in zip(res1["frame_records"], res2["frame_records"]):
        assert r1["camera_id"] == r2["camera_id"]
        assert r1["frame_index"] == r2["frame_index"]
        assert r1["timestamp"] == r2["timestamp"]
        assert r1["source_video"] == r2["source_video"]


# ============================================================================
# 6. FULL DATASET OUTPUT VERIFICATION TESTS
# ============================================================================


def test_frame_index_jsonl_structure_and_records():
    """Verify data/frames/frame_index.jsonl exists and contains valid records for all 11 cameras."""
    index_file = Path("data/frames/frame_index.jsonl")
    if not index_file.exists():
        pytest.skip("data/frames/frame_index.jsonl does not exist yet")

    records = []
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert len(records) == 7287, f"Expected 7287 sampled frames, got {len(records)}"

    required_keys = {"camera_id", "frame_index", "timestamp", "frame_path", "source_video"}
    cameras_seen = set()

    # Native FPS mapping for verification
    fps_map = {f"cam_{i:02d}": 10.0 for i in range(1, 11)}
    fps_map["cam_11"] = 8.0

    for rec in records:
        assert required_keys.issubset(rec.keys()), f"Missing keys in record: {rec}"
        cam_id = rec["camera_id"]
        cameras_seen.add(cam_id)
        frame_idx = rec["frame_index"]
        ts = rec["timestamp"]

        # Verify timestamp matches frame_index / source_fps exactly
        expected_ts = round(frame_idx / fps_map[cam_id], 4)
        assert ts == expected_ts, f"Timestamp mismatch for {cam_id} frame {frame_idx}: {ts} != {expected_ts}"

        # Verify no dynamic timestamp keys
        assert "generated_at" not in rec

        # Verify frame path format
        assert rec["frame_path"] == f"data/frames/{cam_id}/frame_{frame_idx:06d}.jpg"

    assert len(cameras_seen) == 11, "All 11 cameras must be represented in frame_index.jsonl"


def test_sampling_manifest_structure_and_counts():
    """Verify data/frames/sampling_manifest.json exists and contains correct aggregate stats."""
    manifest_path = Path("data/frames/sampling_manifest.json")
    if not manifest_path.exists():
        pytest.skip("data/frames/sampling_manifest.json does not exist yet")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == "1.0"
    assert manifest["sample_fps"] == 3.0
    assert manifest["jpeg_quality"] == 95
    assert manifest["total_cameras"] == 11
    assert manifest["total_source_frames"] == 23798
    assert manifest["total_sampled_frames"] == 7287
    assert manifest["total_frames_disk_bytes"] > 0

    assert "generated_at" not in manifest
    assert len(manifest["cameras"]) == 11

    # Check cam_11 specifically in manifest
    cam_11_entry = next((c for c in manifest["cameras"] if c["camera_id"] == "cam_11"), None)
    assert cam_11_entry is not None
    assert cam_11_entry["source_fps"] == 8.0
    assert cam_11_entry["total_source_frames"] == 1928
    assert cam_11_entry["sampled_frames_count"] == 723


def test_sampled_frames_disk_spot_check():
    """Spot-check sampled frames on disk across different resolutions and FPS."""
    spot_checks = [
        ("cam_01", 0, 1920, 1080),
        ("cam_01", 30, 1920, 1080),
        ("cam_05", 0, 1280, 960),
        ("cam_07", 0, 2560, 1920),
        ("cam_11", 0, 1920, 1080),
        ("cam_11", 24, 1920, 1080),
    ]
    for cam_id, frame_idx, exp_w, exp_h in spot_checks:
        frame_path = Path(f"data/frames/{cam_id}/frame_{frame_idx:06d}.jpg")
        assert frame_path.exists(), f"Sampled frame missing: {frame_path}"
        assert frame_path.stat().st_size > 0, f"Sampled frame empty: {frame_path}"

        img = cv2.imread(str(frame_path))
        assert img is not None, f"Failed to decode {frame_path}"
        h, w = img.shape[:2]
        assert w == exp_w, f"Width mismatch for {frame_path}: {w} != {exp_w}"
        assert h == exp_h, f"Height mismatch for {frame_path}: {h} != {exp_h}"

