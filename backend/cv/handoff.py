"""Stage 8: Person 1 Downstream Handoff & Interface Freeze.

Provides deterministic packaging, manifest generation, and contract verification
for downstream consumption by Person 2 (multimodal embedding and semantic retrieval).
"""

import json
from pathlib import Path
from typing import Any, Optional

from backend.cv.discovery import get_repo_root, normalize_repo_path
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

HANDOFF_MANIFEST_PATH = "data/observations/person1_handoff_manifest.json"


def generate_person1_handoff_manifest(
    root: Optional[Path] = None, output_file: Optional[Path] = None
) -> dict[str, Any]:
    """Generate a deterministic handoff manifest summarizing frozen Person 1 deliverables.

    This manifest contains zero non-deterministic timestamps, zero machine-specific paths,
    and verified dataset counts to provide an unequivocal contract for Person 2.
    """
    if root is None:
        root = get_repo_root()

    if output_file is None:
        output_file = root / HANDOFF_MANIFEST_PATH

    # Execute audit to gather verified ground-truth counts
    audit = run_stage7_integration_audit(root)
    if not audit.passed:
        raise RuntimeError(f"Cannot generate handoff manifest: audit failed with errors: {audit.errors}")

    per_cam_data = []
    for cid in EXPECTED_CAMERAS:
        rec = audit.per_camera_reconciliation[cid]
        per_cam_data.append({
            "camera_id": cid,
            "sampled_frames": rec["sampled_frames"],
            "detections": rec["detections"],
            "tracked_records": rec["tracked_records"],
            "observations": rec["observations"],
            "crops": rec["crops"],
            "unique_trajectories": rec["unique_objects"],
            "source_video": f"data/videos/{cid}/vdo.avi",
            "frame_directory": f"data/frames/{cid}",
            "crop_directory": f"data/crops/{cid}",
        })

    manifest = {
        "project": "ARGUS — Conversational Multi-Camera Video Intelligence",
        "team_id": "HNX26EPS05",
        "stage": "Stage 8 — Person 1 Downstream Handoff & Interface Freeze",
        "role": "Person 1 — Computer Vision & Video Intelligence Engineer",
        "schema_version": "1.0",
        "handoff_status": "FROZEN",
        "dataset_summary": {
            "dataset_name": "CityFlowV2 (AI City Challenge 2022 Track 1)",
            "scenarios": ["S01", "S03"],
            "camera_count": audit.cameras_checked,
            "frame_count": audit.frames_checked,
            "detection_count": audit.detections_checked,
            "track_count": audit.tracks_checked,
            "observation_count": audit.observations_checked,
            "crop_count": audit.crops_checked,
            "trajectory_count": audit.summary["unique_objects_total"],
        },
        "authoritative_artifacts": {
            "observations_file": "data/observations/observations.jsonl",
            "observation_manifest": "data/observations/observation_manifest.json",
            "crop_root_directory": "data/crops",
            "frame_root_directory": "data/frames",
            "video_root_directory": "data/videos",
        },
        "observation_contract": {
            "required_keys": FROZEN_OBSERVATION_KEYS,
            "prohibited_keys": [
                "detection_id",
                "embedding",
                "vector",
                "clip_embedding",
                "dino_embedding",
                "faiss_id",
                "similarity_score",
                "query",
                "ranking_score",
                "database_id",
                "reid_id",
                "global_object_id",
            ],
            "key_count": len(FROZEN_OBSERVATION_KEYS),
            "id_format": "obs_{index:06d}",
            "id_range": "obs_000001 to obs_061039",
            "ordering": ["camera_id", "frame_index", "object_id"],
        },
        "crop_specification": {
            "format": "JPEG",
            "quality": 95,
            "color_space": "RGB",
            "padding": 0,
            "naming_convention": "data/crops/<camera_id>/<observation_id>.jpg",
            "bounding_box_derived": True,
        },
        "contracts": {
            "path_convention": "project-relative POSIX forward slashes (e.g. data/...)",
            "timestamp_definition": "source-video elapsed playback seconds (frame_index / source_video_fps)",
            "object_id_scope": "strictly camera-local independent namespace (obj_camXX_NNNNNN)",
            "cross_camera_reid": "none performed by Person 1; downstream responsibility",
            "embeddings": "none produced by Person 1; downstream responsibility (Person 2)",
        },
        "cameras": per_cam_data,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    m = generate_person1_handoff_manifest()
    print("=" * 65)
    print("STAGE 8 — PERSON 1 DOWNSTREAM HANDOFF MANIFEST GENERATED")
    print("=" * 65)
    print(f"Manifest Path: {HANDOFF_MANIFEST_PATH}")
    print(f"Cameras: {m['dataset_summary']['camera_count']}")
    print(f"Frames: {m['dataset_summary']['frame_count']}")
    print(f"Observations: {m['dataset_summary']['observation_count']}")
    print(f"Crops: {m['dataset_summary']['crop_count']}")
    print(f"Trajectories: {m['dataset_summary']['trajectory_count']}")
    print(f"Status: {m['handoff_status']}")
    print("=" * 65)
