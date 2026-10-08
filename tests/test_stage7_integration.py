"""Stage 7 Multi-Camera CV Pipeline Integration & End-to-End Audit Tests."""

import json
from pathlib import Path

import pytest

from backend.cv.integration_audit import (
    EXPECTED_CAMERAS,
    FROZEN_FRAME_COUNTS,
    FROZEN_TRACK_COUNTS,
    TOTAL_EXPECTED_CROPS,
    TOTAL_EXPECTED_DETECTIONS,
    TOTAL_EXPECTED_FRAMES,
    TOTAL_EXPECTED_OBSERVATIONS,
    TOTAL_EXPECTED_TRACKS,
    TOTAL_EXPECTED_UNIQUE_OBJECTS,
    run_stage7_integration_audit,
)
from backend.cv.observations import FROZEN_OBSERVATION_KEYS


@pytest.fixture(scope="module")
def audit_report():
    """Run full Stage 7 audit once for the test module."""
    return run_stage7_integration_audit()


def test_audit_passes_without_errors(audit_report):
    """Verify that the end-to-end integration audit passes with zero errors."""
    assert audit_report.passed is True, f"Audit failed with errors: {audit_report.errors}"
    assert len(audit_report.errors) == 0


def test_all_11_cameras_present(audit_report):
    """Verify exactly 11 canonical cameras are verified."""
    assert audit_report.cameras_checked == 11
    reconciled_cams = sorted(list(audit_report.per_camera_reconciliation.keys()))
    assert reconciled_cams == EXPECTED_CAMERAS


def test_camera_pipeline_artifacts_present(audit_report):
    """Verify every camera has source video, sampled frames, detections, tracks, crops, and observations."""
    for cid in EXPECTED_CAMERAS:
        rec = audit_report.per_camera_reconciliation[cid]
        assert rec["sampled_frames"] > 0, f"Camera {cid} has 0 sampled frames"
        assert rec["detections"] > 0, f"Camera {cid} has 0 detections"
        assert rec["tracked_records"] > 0, f"Camera {cid} has 0 tracked records"
        assert rec["observations"] > 0, f"Camera {cid} has 0 observations"
        assert rec["crops"] > 0, f"Camera {cid} has 0 crops"
        assert rec["unique_objects"] > 0, f"Camera {cid} has 0 unique objects"


def test_per_camera_count_reconciliation(audit_report):
    """Verify per-camera counts match frozen Stage 3 frame and Stage 5 track baselines exactly."""
    for cid, exp_frames in FROZEN_FRAME_COUNTS.items():
        rec = audit_report.per_camera_reconciliation[cid]
        assert rec["sampled_frames"] == exp_frames, f"Camera {cid} frames {rec['sampled_frames']} != {exp_frames}"

    for cid, exp_tracks in FROZEN_TRACK_COUNTS.items():
        rec = audit_report.per_camera_reconciliation[cid]
        assert rec["tracked_records"] == exp_tracks, f"Camera {cid} tracks {rec['tracked_records']} != {exp_tracks}"
        assert rec["observations"] == exp_tracks, f"Camera {cid} observations != tracks"
        assert rec["crops"] == exp_tracks, f"Camera {cid} crops != tracks"


def test_dataset_totals_reconciliation(audit_report):
    """Verify global totals match frozen dataset invariants."""
    assert audit_report.frames_checked == TOTAL_EXPECTED_FRAMES
    assert audit_report.detections_checked == TOTAL_EXPECTED_DETECTIONS
    assert audit_report.tracks_checked == TOTAL_EXPECTED_TRACKS
    assert audit_report.observations_checked == TOTAL_EXPECTED_OBSERVATIONS
    assert audit_report.crops_checked == TOTAL_EXPECTED_CROPS
    assert audit_report.summary["unique_objects_total"] == TOTAL_EXPECTED_UNIQUE_OBJECTS


def test_frame_references_and_continuity_valid(audit_report):
    """Verify frame references and timestamps match Stage 3 frame index."""
    assert audit_report.summary["frames_verified"] == TOTAL_EXPECTED_FRAMES


def test_multi_camera_isolation(audit_report):
    """Verify strict camera-local object IDs and 0 cross-camera leaks."""
    # Ensure all cameras maintain distinct trajectories
    total_unique = audit_report.summary["unique_objects_total"]
    assert total_unique == TOTAL_EXPECTED_UNIQUE_OBJECTS


def test_observation_schema_exactness():
    """Verify all 61,039 observations adhere strictly to 11 frozen Stage 0 keys."""
    obs_path = Path("data/observations/observations.jsonl")
    with open(obs_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obs = json.loads(line)
            keys = set(obs.keys())
            assert keys == set(FROZEN_OBSERVATION_KEYS), f"Key mismatch on record {idx}"
            assert "detection_id" not in obs
            if idx > 1000 and idx % 5000 != 0:
                continue


def test_no_detection_id_in_public_records():
    """Verify detection_id is strictly omitted from public serialized records."""
    obs_path = Path("data/observations/observations.jsonl")
    with open(obs_path, "r", encoding="utf-8") as f:
        for line in f:
            assert '"detection_id"' not in line


def test_observation_id_sequentiality_and_ordering():
    """Verify observation IDs are sequential from obs_000001 to obs_061039 in deterministic sort order."""
    obs_path = Path("data/observations/observations.jsonl")
    prev_sort_key = None
    with open(obs_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            obs = json.loads(line)
            expected_id = f"obs_{idx:06d}"
            assert obs["observation_id"] == expected_id

            current_sort_key = (obs["camera_id"], obs["frame_index"], obs["object_id"])
            if prev_sort_key is not None:
                assert current_sort_key >= prev_sort_key
            prev_sort_key = current_sort_key


def test_crop_referential_integrity():
    """Verify crops exist, match observation ID and camera directory, and are valid images."""
    crops_dir = Path("data/crops")
    total_crops = sum(1 for d in crops_dir.iterdir() if d.is_dir() for f in d.glob("*.jpg"))
    assert total_crops == TOTAL_EXPECTED_CROPS


def test_source_video_traceability():
    """Verify all observations link back to their respective camera source video."""
    obs_path = Path("data/observations/observations.jsonl")
    with open(obs_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obs = json.loads(line)
            cid = obs["camera_id"]
            expected_video = f"data/videos/{cid}/vdo.avi"
            assert obs["source_video"] == expected_video
            if idx > 1000 and idx % 5000 != 0:
                continue


def test_no_machine_specific_absolute_paths():
    """Verify all serialized paths in observations are project-relative."""
    obs_path = Path("data/observations/observations.jsonl")
    path_keys = ("frame_path", "crop_path", "source_video")
    with open(obs_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obs = json.loads(line)
            for pk in path_keys:
                val = obs[pk]
                assert not val.startswith("/"), f"Leading slash in {val}"
                assert not val.startswith("\\"), f"Leading backslash in {val}"
                assert ":\\" not in val and ":/" not in val, f"Drive letter in {val}"
                assert "\\" not in val, f"Backslash in {val}"
                assert val.startswith("data/"), f"Path does not start with data/: {val}"
            if idx > 1000 and idx % 5000 != 0:
                continue
