"""Stage 7: Multi-Camera CV Pipeline Integration & End-to-End Audit.

Verifies end-to-end coherence, multi-camera isolation, referential integrity,
and deterministic serialization across Stages 1–6 artifacts.
"""

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2

from backend.cv.discovery import get_repo_root, normalize_repo_path
from backend.cv.crops import validate_crop_file
from backend.cv.observations import (
    FROZEN_OBSERVATION_KEYS,
    OBSERVATION_ID_PATTERN,
    validate_observation_record,
)

EXPECTED_CAMERAS = [
    "cam_01",
    "cam_02",
    "cam_03",
    "cam_04",
    "cam_05",
    "cam_06",
    "cam_07",
    "cam_08",
    "cam_09",
    "cam_10",
    "cam_11",
]

FROZEN_FRAME_COUNTS = {
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

FROZEN_TRACK_COUNTS = {
    "cam_01": 3956,
    "cam_02": 6063,
    "cam_03": 5324,
    "cam_04": 4308,
    "cam_05": 5267,
    "cam_06": 4357,
    "cam_07": 6104,
    "cam_08": 5949,
    "cam_09": 1810,
    "cam_10": 5948,
    "cam_11": 11953,
}

TOTAL_EXPECTED_FRAMES = 7287
TOTAL_EXPECTED_DETECTIONS = 68906
TOTAL_EXPECTED_TRACKS = 61039
TOTAL_EXPECTED_OBSERVATIONS = 61039
TOTAL_EXPECTED_CROPS = 61039
TOTAL_EXPECTED_UNIQUE_OBJECTS = 933

OBJECT_ID_PATTERN = re.compile(r"^obj_cam(\d{2})_(\d{6})$")


@dataclass
class AuditReport:
    """Encapsulates Stage 7 audit findings and metrics."""

    passed: bool = True
    audit_runtime_seconds: float = 0.0
    cameras_checked: int = 0
    frames_checked: int = 0
    detections_checked: int = 0
    tracks_checked: int = 0
    observations_checked: int = 0
    crops_checked: int = 0
    errors: list[str] = field(default_factory=list)
    per_camera_reconciliation: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def audit_camera_coverage(root: Path) -> tuple[bool, list[str], dict[str, Any]]:
    """Audit camera coverage across all stages: 11 cameras must be present in all stages."""
    errors: list[str] = []
    inventory_path = root / "data/observations/camera_inventory.json"
    if not inventory_path.exists():
        inventory_path = root / "data/camera_inventory.json"
    if not inventory_path.exists():
        return False, ["Missing camera_inventory.json"], {}

    with open(inventory_path, "r", encoding="utf-8") as f:
        inv = json.load(f)

    raw_cams = inv.get("cameras", [])
    if isinstance(raw_cams, list):
        inv_cameras = sorted([c["camera_id"] for c in raw_cams if "camera_id" in c])
    else:
        inv_cameras = sorted(list(raw_cams.keys()))

    if inv_cameras != EXPECTED_CAMERAS:
        errors.append(f"Inventory cameras {inv_cameras} != expected {EXPECTED_CAMERAS}")

    # Verify per-camera artifacts on disk
    coverage: dict[str, dict[str, bool]] = {}
    for cid in EXPECTED_CAMERAS:
        has_video = (root / f"data/videos/{cid}/vdo.avi").exists()
        has_frames_dir = (root / f"data/frames/{cid}").is_dir()
        has_crops_dir = (root / f"data/crops/{cid}").is_dir()

        coverage[cid] = {
            "video": has_video,
            "frames_dir": has_frames_dir,
            "crops_dir": has_crops_dir,
        }
        if not has_video:
            errors.append(f"Missing video for {cid}")
        if not has_frames_dir:
            errors.append(f"Missing frames dir for {cid}")
        if not has_crops_dir:
            errors.append(f"Missing crops dir for {cid}")

    return len(errors) == 0, errors, coverage


def audit_frame_continuity(
    root: Path,
    frame_index_map: dict[tuple[str, int], dict[str, Any]],
    observations: list[dict[str, Any]],
    inventory: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Verify observation frame references, timestamps, and continuity against Stage 3 frame index."""
    errors: list[str] = []
    raw_cams = inventory.get("cameras", [])
    if isinstance(raw_cams, list):
        inv_cams = {c["camera_id"]: c for c in raw_cams if "camera_id" in c}
    else:
        inv_cams = raw_cams

    invalid_frame_refs = 0
    camera_mismatches = 0
    timestamp_mismatches = 0
    max_frame_exceeded = 0

    for idx, obs in enumerate(observations):
        cid = obs["camera_id"]
        fidx = obs["frame_index"]
        ts = obs["timestamp"]
        fpath = obs["frame_path"]

        # 1. Camera in inventory
        if cid not in inv_cams:
            errors.append(f"Observation {obs['observation_id']} references unknown camera {cid}")
            camera_mismatches += 1
            if len(errors) > 20:
                break
            continue

        # 2. Camera matches frame path prefix
        expected_prefix = f"data/frames/{cid}/"
        if not fpath.startswith(expected_prefix):
            errors.append(
                f"Observation {obs['observation_id']} frame_path '{fpath}' does not match camera '{cid}'"
            )
            camera_mismatches += 1

        # 3. Lookup in Stage 3 frame index
        f_key = (cid, fidx)
        if f_key not in frame_index_map:
            errors.append(f"Observation {obs['observation_id']} references unindexed frame {f_key}")
            invalid_frame_refs += 1
        else:
            stage3_frame = frame_index_map[f_key]
            # Verify timestamp matches Stage 3
            if abs(ts - stage3_frame["timestamp"]) > 1e-4:
                errors.append(
                    f"Observation {obs['observation_id']} timestamp {ts} != Stage 3 timestamp {stage3_frame['timestamp']}"
                )
                timestamp_mismatches += 1

        # 4. Check frame index within video duration
        cam_meta = inv_cams.get(cid, {})
        total_frames = cam_meta.get("frame_count", 0)
        if total_frames > 0 and fidx >= total_frames:
            errors.append(
                f"Observation {obs['observation_id']} frame_index {fidx} exceeds video total {total_frames}"
            )
            max_frame_exceeded += 1

        if len(errors) > 20:
            errors.append("Too many frame continuity errors, truncating...")
            break

    total_failures = invalid_frame_refs + camera_mismatches + timestamp_mismatches + max_frame_exceeded
    return total_failures == 0, errors


def audit_multi_camera_isolation(observations: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Verify that object IDs are strictly camera-local and have zero cross-camera leakage."""
    errors: list[str] = []
    object_cameras: dict[str, set[str]] = {}

    for obs in observations:
        oid = obs["object_id"]
        cid = obs["camera_id"]

        # Parse camera prefix from object ID
        match = OBJECT_ID_PATTERN.match(oid)
        if not match:
            errors.append(f"Object ID '{oid}' does not match strict camera-local pattern")
            continue

        cam_num = match.group(1)
        expected_cid = f"cam_{cam_num}"
        if cid != expected_cid:
            errors.append(
                f"Object ID '{oid}' has camera prefix '{expected_cid}' but is tagged with camera '{cid}'"
            )

        if oid not in object_cameras:
            object_cameras[oid] = set()
        object_cameras[oid].add(cid)

    # Check for any object ID spanning multiple cameras
    cross_camera_objects = {oid: cams for oid, cams in object_cameras.items() if len(cams) > 1}
    if cross_camera_objects:
        for oid, cams in list(cross_camera_objects.items())[:10]:
            errors.append(f"Cross-camera leak detected: Object ID '{oid}' appears in multiple cameras {cams}")

    return len(errors) == 0, errors


def audit_observation_id_integrity(observations: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Verify that observation IDs are sequentially numbered, unique, and deterministically sorted."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    prev_sort_key: Optional[tuple[str, int, str]] = None

    for idx, obs in enumerate(observations, start=1):
        obs_id = obs["observation_id"]

        # 1. Pattern check
        if not OBSERVATION_ID_PATTERN.match(obs_id):
            errors.append(f"Observation ID '{obs_id}' does not match pattern obs_\\d{{6}}")

        # 2. Sequential check
        expected_id = f"obs_{idx:06d}"
        if obs_id != expected_id:
            errors.append(f"Observation at position {idx} has ID '{obs_id}', expected '{expected_id}'")

        # 3. Uniqueness check
        if obs_id in seen_ids:
            errors.append(f"Duplicate observation ID found: '{obs_id}'")
        seen_ids.add(obs_id)

        # 4. Deterministic sorting check: (camera_id, frame_index, object_id)
        current_sort_key = (obs["camera_id"], obs["frame_index"], obs["object_id"])
        if prev_sort_key is not None and current_sort_key < prev_sort_key:
            errors.append(
                f"Observation ordering violation at {obs_id}: {current_sort_key} < {prev_sort_key}"
            )
        prev_sort_key = current_sort_key

        if len(errors) > 20:
            errors.append("Too many observation ID errors, aborting check...")
            break

    return len(errors) == 0, errors


def audit_crop_traceability(root: Path, observations: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Verify that crop files exist, are readable, have non-zero dimensions, and correspond to obs_id."""
    errors: list[str] = []
    missing_crops = 0
    invalid_crops = 0

    # Spot check thoroughly: verify 100% file existence, spot-check decodability on sample
    for idx, obs in enumerate(observations):
        crop_rel = obs["crop_path"]
        crop_abs = root / crop_rel

        # Verify path matches convention data/crops/<camera_id>/<obs_id>.jpg
        expected_rel = f"data/crops/{obs['camera_id']}/{obs['observation_id']}.jpg"
        if crop_rel != expected_rel:
            errors.append(f"Crop path mismatch for {obs['observation_id']}: '{crop_rel}' != '{expected_rel}'")

        if not crop_abs.exists():
            missing_crops += 1
            if missing_crops <= 5:
                errors.append(f"Missing crop on disk: {crop_rel}")
            continue

        # Spot check decoding on every 200th crop to keep audit lightweight yet rigorous
        if idx % 200 == 0:
            is_valid, dims, err = validate_crop_file(crop_abs)
            if not is_valid or dims is None or dims[0] <= 0 or dims[1] <= 0:
                invalid_crops += 1
                errors.append(f"Invalid crop image {crop_rel}: {err}")

    if missing_crops > 0:
        errors.append(f"Total missing crops: {missing_crops}")
    if invalid_crops > 0:
        errors.append(f"Total invalid crops: {invalid_crops}")

    return len(errors) == 0, errors


def audit_source_video_traceability(root: Path, observations: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Verify that every observation traces back to its valid camera source video."""
    errors: list[str] = []
    for obs in observations:
        cid = obs["camera_id"]
        vpath = obs["source_video"]
        expected_video = f"data/videos/{cid}/vdo.avi"
        if vpath != expected_video:
            errors.append(
                f"Observation {obs['observation_id']} points to source video '{vpath}', expected '{expected_video}'"
            )
            if len(errors) > 10:
                break

    return len(errors) == 0, errors


def audit_path_formatting(observations: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Verify that all serialized paths are strictly project-relative and contain no absolute/machine paths."""
    errors: list[str] = []
    path_keys = ("frame_path", "crop_path", "source_video")

    for obs in observations:
        for pk in path_keys:
            val = obs[pk]
            # Disallow absolute Windows or POSIX indicators
            if ":\\" in val or ":/" in val or val.startswith("/") or val.startswith("\\"):
                errors.append(f"Absolute path found in {obs['observation_id']}[{pk}]: {val}")
            if "\\" in val:
                errors.append(f"Backslash found in repo path in {obs['observation_id']}[{pk}]: {val}")
            if not val.startswith("data/"):
                errors.append(f"Path does not start with 'data/' in {obs['observation_id']}[{pk}]: {val}")

        if len(errors) > 20:
            errors.append("Too many path format errors, aborting check...")
            break

    return len(errors) == 0, errors


def audit_contract_compliance(observations: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Audit strict Stage 0 schema compliance across all observation records."""
    errors: list[str] = []
    for idx, obs in enumerate(observations):
        is_valid, err = validate_observation_record(obs)
        if not is_valid:
            errors.append(f"Record {idx} ({obs.get('observation_id')}) schema error: {err}")
            if len(errors) > 20:
                break

    return len(errors) == 0, errors


def run_stage7_integration_audit(root: Optional[Path] = None) -> AuditReport:
    """Execute complete Stage 7 end-to-end multi-camera integration audit."""
    if root is None:
        root = get_repo_root()

    t_start = time.time()
    report = AuditReport()
    all_errors: list[str] = []

    # 1. Camera Coverage
    cov_ok, cov_errs, cov_details = audit_camera_coverage(root)
    if not cov_ok:
        all_errors.extend(cov_errs)
    report.cameras_checked = len(EXPECTED_CAMERAS)

    # Load inventory
    inv_path = root / "data/observations/camera_inventory.json"
    if not inv_path.exists():
        inv_path = root / "data/camera_inventory.json"
    with open(inv_path, "r", encoding="utf-8") as f:
        inventory = json.load(f)

    # 2. Load Frame Index
    frame_index_path = root / "data/frames/frame_index.jsonl"
    frame_index_map: dict[tuple[str, int], dict[str, Any]] = {}
    frames_per_camera: dict[str, int] = {c: 0 for c in EXPECTED_CAMERAS}

    with open(frame_index_path, "r", encoding="utf-8") as f:
        for line in f:
            f_record = json.loads(line)
            cid = f_record["camera_id"]
            fidx = f_record["frame_index"]
            frame_index_map[(cid, fidx)] = f_record
            if cid in frames_per_camera:
                frames_per_camera[cid] += 1

    report.frames_checked = len(frame_index_map)
    if report.frames_checked != TOTAL_EXPECTED_FRAMES:
        all_errors.append(f"Frame count {report.frames_checked} != expected {TOTAL_EXPECTED_FRAMES}")

    for cid, exp_cnt in FROZEN_FRAME_COUNTS.items():
        if frames_per_camera.get(cid, 0) != exp_cnt:
            all_errors.append(
                f"Camera {cid} frame count {frames_per_camera.get(cid, 0)} != frozen {exp_cnt}"
            )

    # 3. Load Detections counts
    detections_path = root / "data/detections/detections.jsonl"
    detections_per_camera: dict[str, int] = {c: 0 for c in EXPECTED_CAMERAS}
    total_detections = 0
    with open(detections_path, "r", encoding="utf-8") as f:
        for line in f:
            det = json.loads(line)
            cid = det["camera_id"]
            if cid in detections_per_camera:
                detections_per_camera[cid] += 1
            total_detections += 1

    report.detections_checked = total_detections
    if total_detections != TOTAL_EXPECTED_DETECTIONS:
        all_errors.append(f"Total detections {total_detections} != expected {TOTAL_EXPECTED_DETECTIONS}")

    # 4. Load Tracks
    tracks_path = root / "data/tracks/tracks.jsonl"
    tracks_per_camera: dict[str, int] = {c: 0 for c in EXPECTED_CAMERAS}
    unique_objects_per_camera: dict[str, set[str]] = {c: set() for c in EXPECTED_CAMERAS}
    total_tracks = 0
    with open(tracks_path, "r", encoding="utf-8") as f:
        for line in f:
            trk = json.loads(line)
            cid = trk["camera_id"]
            if cid in tracks_per_camera:
                tracks_per_camera[cid] += 1
                unique_objects_per_camera[cid].add(trk["object_id"])
            total_tracks += 1

    report.tracks_checked = total_tracks
    if total_tracks != TOTAL_EXPECTED_TRACKS:
        all_errors.append(f"Total tracks {total_tracks} != expected {TOTAL_EXPECTED_TRACKS}")

    for cid, exp_trk in FROZEN_TRACK_COUNTS.items():
        if tracks_per_camera.get(cid, 0) != exp_trk:
            all_errors.append(
                f"Camera {cid} track count {tracks_per_camera.get(cid, 0)} != frozen {exp_trk}"
            )

    # 5. Load Observations
    obs_path = root / "data/observations/observations.jsonl"
    observations: list[dict[str, Any]] = []
    obs_per_camera: dict[str, int] = {c: 0 for c in EXPECTED_CAMERAS}
    with open(obs_path, "r", encoding="utf-8") as f:
        for line in f:
            obs = json.loads(line)
            observations.append(obs)
            cid = obs["camera_id"]
            if cid in obs_per_camera:
                obs_per_camera[cid] += 1

    report.observations_checked = len(observations)
    if report.observations_checked != TOTAL_EXPECTED_OBSERVATIONS:
        all_errors.append(
            f"Total observations {report.observations_checked} != expected {TOTAL_EXPECTED_OBSERVATIONS}"
        )

    # 6. Count Crops on Disk
    crops_dir = root / "data/crops"
    crops_per_camera: dict[str, int] = {c: 0 for c in EXPECTED_CAMERAS}
    total_crops_on_disk = 0
    for cid in EXPECTED_CAMERAS:
        cam_crops_dir = crops_dir / cid
        if cam_crops_dir.is_dir():
            c_cnt = sum(1 for f in cam_crops_dir.glob("*.jpg"))
            crops_per_camera[cid] = c_cnt
            total_crops_on_disk += c_cnt

    report.crops_checked = total_crops_on_disk
    if total_crops_on_disk != TOTAL_EXPECTED_CROPS:
        all_errors.append(f"Total crops {total_crops_on_disk} != expected {TOTAL_EXPECTED_CROPS}")

    # 7. Audit Frame Continuity
    fc_ok, fc_errs = audit_frame_continuity(root, frame_index_map, observations, inventory)
    if not fc_ok:
        all_errors.extend(fc_errs)

    # 8. Audit Multi-Camera Isolation
    iso_ok, iso_errs = audit_multi_camera_isolation(observations)
    if not iso_ok:
        all_errors.extend(iso_errs)

    # 9. Audit Observation ID Integrity
    id_ok, id_errs = audit_observation_id_integrity(observations)
    if not id_ok:
        all_errors.extend(id_errs)

    # 10. Audit Crop Traceability
    crp_ok, crp_errs = audit_crop_traceability(root, observations)
    if not crp_ok:
        all_errors.extend(crp_errs)

    # 11. Audit Source Video Traceability
    vid_ok, vid_errs = audit_source_video_traceability(root, observations)
    if not vid_ok:
        all_errors.extend(vid_errs)

    # 12. Audit Path Formatting (no absolute paths)
    path_ok, path_errs = audit_path_formatting(observations)
    if not path_ok:
        all_errors.extend(path_errs)

    # 13. Audit Contract Compliance (exact 11 keys, no detection_id)
    contract_ok, contract_errs = audit_contract_compliance(observations)
    if not contract_ok:
        all_errors.extend(contract_errs)

    # 14. Compile Per-Camera Reconciliation Table
    reconciliation: dict[str, dict[str, Any]] = {}
    total_unique_objects = sum(len(objs) for objs in unique_objects_per_camera.values())

    for cid in EXPECTED_CAMERAS:
        reconciliation[cid] = {
            "camera_id": cid,
            "sampled_frames": frames_per_camera[cid],
            "detections": detections_per_camera[cid],
            "tracked_records": tracks_per_camera[cid],
            "observations": obs_per_camera[cid],
            "crops": crops_per_camera[cid],
            "unique_objects": len(unique_objects_per_camera[cid]),
        }

    report.per_camera_reconciliation = reconciliation
    report.errors = all_errors
    report.passed = len(all_errors) == 0

    t_end = time.time()
    report.audit_runtime_seconds = round(t_end - t_start, 3)

    report.summary = {
        "status": "PASSED" if report.passed else "FAILED",
        "cameras_verified": report.cameras_checked,
        "frames_verified": report.frames_checked,
        "detections_verified": report.detections_checked,
        "tracks_verified": report.tracks_checked,
        "observations_verified": report.observations_checked,
        "crops_verified": report.crops_checked,
        "unique_objects_total": total_unique_objects,
        "error_count": len(all_errors),
        "runtime_seconds": report.audit_runtime_seconds,
    }

    return report


if __name__ == "__main__":
    rep = run_stage7_integration_audit()
    print("=" * 70)
    print("STAGE 7 — MULTI-CAMERA CV PIPELINE INTEGRATION AUDIT")
    print("=" * 70)
    print(f"Status: {rep.summary['status']}")
    print(f"Audit Runtime: {rep.audit_runtime_seconds}s")
    print(f"Cameras Checked: {rep.cameras_checked}")
    print(f"Frames Checked: {rep.frames_checked}")
    print(f"Detections Checked: {rep.detections_checked}")
    print(f"Tracks Checked: {rep.tracks_checked}")
    print(f"Observations Checked: {rep.observations_checked}")
    print(f"Crops Checked: {rep.crops_checked}")
    print(f"Unique Trajectories: {rep.summary['unique_objects_total']}")
    print(f"Errors Found: {len(rep.errors)}")
    if rep.errors:
        print("\nERRORS:")
        for e in rep.errors[:20]:
            print(f"  - {e}")
    else:
        print("\nAll integration and multi-camera isolation invariants passed!")
    print("=" * 70)
