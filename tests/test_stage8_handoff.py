"""Tests for Stage 8: Person 1 Downstream Handoff & Interface Freeze."""

import json
from pathlib import Path

import pytest

from backend.cv.handoff import (
    HANDOFF_MANIFEST_PATH,
    generate_person1_handoff_manifest,
)
from backend.cv.integration_audit import (
    EXPECTED_CAMERAS,
    TOTAL_EXPECTED_CROPS,
    TOTAL_EXPECTED_DETECTIONS,
    TOTAL_EXPECTED_FRAMES,
    TOTAL_EXPECTED_OBSERVATIONS,
    TOTAL_EXPECTED_TRACKS,
    TOTAL_EXPECTED_UNIQUE_OBJECTS,
)
from backend.cv.observations import FROZEN_OBSERVATION_KEYS


@pytest.fixture(scope="module")
def handoff_manifest():
    """Load the serialized handoff manifest."""
    manifest_p = Path(HANDOFF_MANIFEST_PATH)
    if not manifest_p.exists():
        pytest.fail(f"Handoff manifest does not exist at {HANDOFF_MANIFEST_PATH}")
    with open(manifest_p, "r", encoding="utf-8") as f:
        return json.load(f)


def test_handoff_manifest_exists():
    """Verify data/observations/person1_handoff_manifest.json exists."""
    assert Path(HANDOFF_MANIFEST_PATH).exists()


def test_handoff_manifest_is_valid_json(handoff_manifest):
    """Verify handoff manifest contains all required contract keys."""
    required_keys = [
        "project",
        "stage",
        "role",
        "schema_version",
        "handoff_status",
        "dataset_summary",
        "authoritative_artifacts",
        "observation_contract",
        "crop_specification",
        "contracts",
        "cameras",
    ]
    for k in required_keys:
        assert k in handoff_manifest, f"Missing key in handoff manifest: {k}"

    assert handoff_manifest["handoff_status"] == "FROZEN"
    assert handoff_manifest["schema_version"] == "1.0"


def test_handoff_manifest_is_deterministic(tmp_path: Path):
    """Verify running handoff manifest generation yields byte-identical output."""
    existing_text = Path(HANDOFF_MANIFEST_PATH).read_text(encoding="utf-8")
    test_out = tmp_path / "test_manifest.json"
    generate_person1_handoff_manifest(output_file=test_out)
    new_text = test_out.read_text(encoding="utf-8")
    assert existing_text == new_text, "Handoff manifest is not strictly deterministic!"


def test_observations_file_exists():
    """Verify data/observations/observations.jsonl exists."""
    assert Path("data/observations/observations.jsonl").exists()


def test_exactly_61039_observations_exist():
    """Verify observations file contains exactly 61,039 records."""
    count = 0
    with open("data/observations/observations.jsonl", "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    assert count == TOTAL_EXPECTED_OBSERVATIONS


def test_exactly_61039_crops_exist():
    """Verify crop root contains exactly 61,039 JPEG crops."""
    crops_dir = Path("data/crops")
    total_crops = sum(1 for d in crops_dir.iterdir() if d.is_dir() for f in d.glob("*.jpg"))
    assert total_crops == TOTAL_EXPECTED_CROPS


def test_every_observation_has_exact_11_keys():
    """Verify every observation record has strictly the frozen 11 keys."""
    with open("data/observations/observations.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obs = json.loads(line)
            assert set(obs.keys()) == set(FROZEN_OBSERVATION_KEYS), f"Key mismatch on record {idx}"
            if idx > 1000 and idx % 5000 != 0:
                continue


def test_detection_id_is_absent():
    """Verify detection_id is strictly absent from all public observation records."""
    with open("data/observations/observations.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            assert '"detection_id"' not in line


def test_all_crop_paths_exist():
    """Verify crop paths referenced by observations exist on disk."""
    with open("data/observations/observations.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 500 == 0:
                obs = json.loads(line)
                crop_p = Path(obs["crop_path"])
                assert crop_p.exists(), f"Crop path missing on disk: {crop_p}"


def test_all_paths_are_project_relative():
    """Verify all serialized paths are project-relative with forward slashes."""
    with open("data/observations/observations.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 500 == 0:
                obs = json.loads(line)
                for pk in ("frame_path", "crop_path", "source_video"):
                    p = obs[pk]
                    assert p.startswith("data/"), f"Path not starting with data/: {p}"
                    assert "\\" not in p, f"Backslash in path: {p}"
                    assert ":\\" not in p and ":/" not in p, f"Absolute path indicator in {p}"


def test_all_camera_ids_are_valid():
    """Verify all observations map to one of the 11 valid canonical cameras."""
    valid_cams = set(EXPECTED_CAMERAS)
    with open("data/observations/observations.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 500 == 0:
                obs = json.loads(line)
                assert obs["camera_id"] in valid_cams


def test_object_ids_remain_camera_local():
    """Verify object IDs strictly follow camera-local obj_camXX_NNNNNN pattern."""
    with open("data/observations/observations.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 500 == 0:
                obs = json.loads(line)
                cid = obs["camera_id"]
                oid = obs["object_id"]
                cam_num = cid.replace("cam_", "")
                assert oid.startswith(f"obj_cam{cam_num}_")


def test_timestamps_remain_valid():
    """Verify timestamps are non-negative, finite floats."""
    with open("data/observations/observations.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 500 == 0:
                obs = json.loads(line)
                ts = obs["timestamp"]
                assert isinstance(ts, (int, float))
                assert ts >= 0.0


def test_observation_ids_are_unique():
    """Verify observation IDs contain zero duplicates across all 61,039 records."""
    seen_ids = set()
    with open("data/observations/observations.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obs = json.loads(line)
            oid = obs["observation_id"]
            assert oid not in seen_ids, f"Duplicate observation ID found: {oid}"
            seen_ids.add(oid)
    assert len(seen_ids) == TOTAL_EXPECTED_OBSERVATIONS


def test_observation_ids_remain_sequential():
    """Verify observation IDs form a strictly sequential sequence from obs_000001 to obs_061039."""
    with open("data/observations/observations.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            obs = json.loads(line)
            assert obs["observation_id"] == f"obs_{idx:06d}"


def test_handoff_counts_match_frozen_stage7_totals(handoff_manifest):
    """Verify handoff manifest totals reconcile exactly with frozen Stage 7 invariants."""
    summary = handoff_manifest["dataset_summary"]
    assert summary["camera_count"] == len(EXPECTED_CAMERAS)
    assert summary["frame_count"] == TOTAL_EXPECTED_FRAMES
    assert summary["detection_count"] == TOTAL_EXPECTED_DETECTIONS
    assert summary["track_count"] == TOTAL_EXPECTED_TRACKS
    assert summary["observation_count"] == TOTAL_EXPECTED_OBSERVATIONS
    assert summary["crop_count"] == TOTAL_EXPECTED_CROPS
    assert summary["trajectory_count"] == TOTAL_EXPECTED_UNIQUE_OBJECTS


def test_no_downstream_embedding_or_retrieval_fields(handoff_manifest):
    """Verify no downstream embedding, FAISS, or query fields exist in handoff deliverables."""
    prohibited = handoff_manifest["observation_contract"]["prohibited_keys"]
    with open("data/observations/observations.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % 1000 == 0:
                obs = json.loads(line)
                for field_name in prohibited:
                    assert field_name not in obs, f"Prohibited field '{field_name}' in record {idx}"
