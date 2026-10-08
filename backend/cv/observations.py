"""Person 1 Stage 6: Object Crop Generation and Observation Record Construction.

Consumes Stage 5 tracking trajectories, extracts deterministic object crops,
and serializes standardized observation records according to the frozen Stage 0 contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2

from backend.cv.crops import (
    DEFAULT_CROP_QUALITY,
    compute_safe_crop_coordinates,
    extract_and_save_crop,
    validate_crop_file,
)
from backend.cv.discovery import get_repo_root, normalize_repo_path

logger = logging.getLogger("argus.cv.observations")

OBSERVATION_ID_PATTERN = re.compile(r"^obs_\d{6}$")

FROZEN_OBSERVATION_KEYS = [
    "observation_id",
    "camera_id",
    "timestamp",
    "frame_index",
    "object_id",
    "object_type",
    "confidence",
    "bbox",
    "frame_path",
    "crop_path",
    "source_video",
]


@dataclass
class ObservationConfig:
    """Configuration for Stage 6 observation and crop generation."""

    tracks_file: str = "data/tracks/tracks.jsonl"
    frame_index_path: str = "data/frames/frame_index.jsonl"
    output_dir: str = "data/observations"
    crops_dir: str = "data/crops"
    observations_file: str = "data/observations/observations.jsonl"
    manifest_file: str = "data/observations/observation_manifest.json"
    jpeg_quality: int = DEFAULT_CROP_QUALITY
    num_workers: int = 4

    def __post_init__(self) -> None:
        """Validate observation configuration."""
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality must be in [1, 100], got {self.jpeg_quality}")
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")


def format_observation_id(numeric_id: int) -> str:
    """Format deterministic observation identifier.

    Example: 1 -> 'obs_000001'
    """
    return f"obs_{numeric_id:06d}"


def validate_observation_record(record: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate all fields and constraints of an individual observation record."""
    if "detection_id" in record:
        return False, "detection_id is forbidden in public observation schema (Stage 0 contract violation)"

    extra_keys = set(record.keys()) - set(FROZEN_OBSERVATION_KEYS)
    if extra_keys:
        return False, f"Unexpected extra keys in observation record: {sorted(extra_keys)}"

    for key in FROZEN_OBSERVATION_KEYS:
        if key not in record:
            return False, f"Missing required observation key: '{key}'"

    obs_id = record["observation_id"]
    if not isinstance(obs_id, str) or not OBSERVATION_ID_PATTERN.match(obs_id):
        return False, f"Invalid observation_id format: '{obs_id}'"

    if not isinstance(record["camera_id"], str) or not record["camera_id"].startswith("cam_"):
        return False, f"Invalid camera_id: '{record['camera_id']}'"

    if not isinstance(record["frame_index"], int) or record["frame_index"] < 0:
        return False, f"Invalid frame_index: {record['frame_index']}"

    ts = record["timestamp"]
    if not isinstance(ts, (int, float)) or math.isnan(ts) or math.isinf(ts) or ts < 0:
        return False, f"Invalid timestamp: {ts}"

    conf = record["confidence"]
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        return False, f"Invalid confidence: {conf}"

    bbox = record["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False, f"bbox must be 4 floats, got {bbox}"
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        return False, f"Degenerate bbox coordinates: {bbox}"
    if bbox[0] < 0.0 or bbox[1] < 0.0:
        return False, f"Negative bbox coordinates: {bbox}"

    for path_key in ("frame_path", "crop_path", "source_video"):
        val = record[path_key]
        if not isinstance(val, str) or not val.strip():
            return False, f"Empty or invalid path for '{path_key}': {val}"

    return True, None


def _process_frame_crops(args: tuple[Path, str, list[dict[str, Any]], int]) -> list[tuple[str, bool, Optional[str]]]:
    """Worker helper to decode a frame once and save all its observation crops.

    Args:
        args: Tuple of (abs_frame_path, repo_relative_frame_path, list_of_obs_metadata, jpeg_quality).

    Returns:
        List of (observation_id, success, error_message).
    """
    abs_frame_path, rel_frame_path, obs_list, quality = args
    results = []

    if not abs_frame_path.exists():
        for item in obs_list:
            results.append((item["observation_id"], False, f"Frame not found: {abs_frame_path}"))
        return results

    frame_img = cv2.imread(str(abs_frame_path))
    if frame_img is None or frame_img.size == 0:
        for item in obs_list:
            results.append((item["observation_id"], False, f"Failed to decode frame: {abs_frame_path}"))
        return results

    img_h, img_w = frame_img.shape[:2]
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    for item in obs_list:
        obs_id = item["observation_id"]
        abs_crop_path = item["abs_crop_path"]
        bbox = item["bbox"]

        x1_int, y1_int, x2_int, y2_int = compute_safe_crop_coordinates(bbox, img_w, img_h)
        crop = frame_img[y1_int:y2_int, x1_int:x2_int]

        if crop.shape[0] <= 0 or crop.shape[1] <= 0:
            results.append((obs_id, False, f"Zero area crop for {obs_id}"))
            continue

        abs_crop_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(abs_crop_path), crop, encode_params)
        if not ok or not abs_crop_path.exists():
            results.append((obs_id, False, f"Failed to write crop {abs_crop_path}"))
        else:
            results.append((obs_id, True, None))

    return results


def run_stage6_observations(
    tracks_file: Path | str = "data/tracks/tracks.jsonl",
    frame_index_path: Path | str = "data/frames/frame_index.jsonl",
    output_dir: Path | str = "data/observations",
    crops_dir: Path | str = "data/crops",
    jpeg_quality: int = DEFAULT_CROP_QUALITY,
    num_workers: int = 4,
    repo_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Execute Stage 6 object crop generation and observation record construction.

    Args:
        tracks_file: Path to Stage 5 tracking file.
        frame_index_path: Path to Stage 3 authoritative frame index.
        output_dir: Output directory for observation records and manifest.
        crops_dir: Root directory for generated crop images.
        jpeg_quality: JPEG compression quality setting (1-100).
        num_workers: Number of worker threads for parallel frame crop generation.
        repo_root: Optional repository root for path resolution.

    Returns:
        Summary manifest dictionary.
    """
    root = Path(repo_root).resolve() if repo_root else get_repo_root()
    config = ObservationConfig(
        tracks_file=str(tracks_file),
        frame_index_path=str(frame_index_path),
        output_dir=str(output_dir),
        crops_dir=str(crops_dir),
        observations_file=f"{output_dir}/observations.jsonl".replace("\\", "/"),
        manifest_file=f"{output_dir}/observation_manifest.json".replace("\\", "/"),
        jpeg_quality=jpeg_quality,
        num_workers=num_workers,
    )

    in_tracks_path = root / config.tracks_file
    if not in_tracks_path.exists():
        raise FileNotFoundError(f"Stage 5 tracks file not found: {in_tracks_path}")

    # 1. Load Stage 5 tracking records
    raw_track_records = []
    with open(in_tracks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_track_records.append(json.loads(line))

    total_tracked_input = len(raw_track_records)
    if total_tracked_input == 0:
        raise RuntimeError("No tracking records found in input file")

    # 2. Sort deterministically by (camera_id, frame_index, object_id)
    raw_track_records.sort(key=lambda r: (r["camera_id"], r["frame_index"], r["object_id"]))

    # 3. Assign sequential observation IDs and assign crop paths
    observations: list[dict[str, Any]] = []
    frame_grouped_tasks: dict[str, list[dict[str, Any]]] = {}

    for idx, trk in enumerate(raw_track_records, start=1):
        obs_id = format_observation_id(idx)
        cid = trk["camera_id"]
        rel_crop_path = f"{config.crops_dir}/{cid}/{obs_id}.jpg".replace("\\", "/")
        abs_crop_path = root / rel_crop_path

        obs_record = {
            "observation_id": obs_id,
            "camera_id": cid,
            "timestamp": trk["timestamp"],
            "frame_index": trk["frame_index"],
            "object_id": trk["object_id"],
            "object_type": trk["object_type"],
            "confidence": trk["confidence"],
            "bbox": trk["bbox"],
            "frame_path": trk["frame_path"],
            "crop_path": rel_crop_path,
            "source_video": trk["source_video"],
        }
        observations.append(obs_record)

        frame_p = trk["frame_path"]
        if frame_p not in frame_grouped_tasks:
            frame_grouped_tasks[frame_p] = []
        frame_grouped_tasks[frame_p].append({
            "observation_id": obs_id,
            "bbox": trk["bbox"],
            "abs_crop_path": abs_crop_path,
        })

    print("=" * 65)
    print("STAGE 6 — OBJECT CROP GENERATION & OBSERVATION CONSTRUCTION")
    print(f"Input Tracks: {normalize_repo_path(in_tracks_path, root)} ({total_tracked_input} records)")
    print(f"Unique Frames Referenced: {len(frame_grouped_tasks)}")
    print(f"Target Observations: {len(observations)} | Workers: {config.num_workers}")
    print(f"JPEG Quality: {config.jpeg_quality} | Crops Dir: {config.crops_dir}")
    print("=" * 65)

    # 4. Generate crops in parallel across unique frames
    worker_args = [
        (root / fp, fp, items, config.jpeg_quality)
        for fp, items in frame_grouped_tasks.items()
    ]

    t_crop_start = time.time()
    failed_crops = 0

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        for res_batch in executor.map(_process_frame_crops, worker_args):
            for obs_id, success, err_msg in res_batch:
                if not success:
                    logger.error(f"Crop generation failed for {obs_id}: {err_msg}")
                    failed_crops += 1

    t_crop_end = time.time()
    crop_duration = round(t_crop_end - t_crop_start, 3)
    crop_rate = round(len(observations) / crop_duration, 1) if crop_duration > 0 else 0
    print(f"Crop generation complete: {len(observations)} crops in {crop_duration}s ({crop_rate} crops/sec).")

    # 5. Serialize observations.jsonl
    out_dir = root / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_file_path = root / config.observations_file
    manifest_file_path = root / config.manifest_file

    observations_per_camera: dict[str, int] = {}
    crops_per_camera: dict[str, int] = {}
    frames_per_camera: dict[str, set[int]] = {}

    failed_observations = 0
    with open(obs_file_path, "w", encoding="utf-8") as out_f:
        for obs in observations:
            is_valid, err = validate_observation_record(obs)
            if not is_valid:
                logger.error(f"Invalid observation record {obs.get('observation_id')}: {err}")
                failed_observations += 1
                continue

            out_f.write(json.dumps(obs, ensure_ascii=False) + "\n")

            cid = obs["camera_id"]
            observations_per_camera[cid] = observations_per_camera.get(cid, 0) + 1
            crops_per_camera[cid] = crops_per_camera.get(cid, 0) + 1
            if cid not in frames_per_camera:
                frames_per_camera[cid] = set()
            frames_per_camera[cid].add(obs["frame_index"])

    # 6. Generate summary manifest
    all_camera_ids = sorted(observations_per_camera.keys())
    frames_referenced_count = {cid: len(frames_per_camera[cid]) for cid in all_camera_ids}

    manifest: dict[str, Any] = {
        "stage": "Stage 6 — Object Crop Generation & Observation Record Construction",
        "schema_version": "1.0",
        "input_track_file": normalize_repo_path(in_tracks_path, root),
        "camera_count": len(all_camera_ids),
        "camera_ids": all_camera_ids,
        "input_tracked_records": total_tracked_input,
        "output_observation_records": len(observations) - failed_observations,
        "crop_count": len(observations) - failed_crops,
        "frames_referenced": len(frame_grouped_tasks),
        "observations_per_camera": observations_per_camera,
        "crops_per_camera": crops_per_camera,
        "frames_referenced_per_camera": frames_referenced_count,
        "crop_configuration": {
            "jpeg_quality": config.jpeg_quality,
            "format": "jpg",
            "padding": 0,
        },
        "validation_results": {
            "failed_crops": failed_crops,
            "failed_observations": failed_observations,
            "verified_integrity": (failed_crops == 0 and failed_observations == 0),
        },
        "performance": {
            "runtime_seconds": crop_duration,
            "observations_per_second": crop_rate,
            "crops_per_second": crop_rate,
        },
        "observations_file": normalize_repo_path(obs_file_path, root),
        "crops_directory": config.crops_dir,
    }

    with open(manifest_file_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print(f"STAGE 6 COMPLETE: {len(observations)} observations and crops generated.")
    print(f"Failed crops: {failed_crops} | Failed observations: {failed_observations}")
    print(f"Observations JSONL: {normalize_repo_path(obs_file_path, root)}")
    print(f"Manifest JSON: {normalize_repo_path(manifest_file_path, root)}")
    print("=" * 65)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person 1 Stage 6 Crop & Observation Generation")
    parser.add_argument("--tracks", default="data/tracks/tracks.jsonl", help="Input Stage 5 tracks file")
    parser.add_argument("--frame-index", default="data/frames/frame_index.jsonl", help="Stage 3 frame index")
    parser.add_argument("--output-dir", default="data/observations", help="Observations output directory")
    parser.add_argument("--crops-dir", default="data/crops", help="Crops output directory")
    parser.add_argument("--quality", type=int, default=DEFAULT_CROP_QUALITY, help="JPEG quality [1-100]")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads for crop generation")
    args = parser.parse_args()

    run_stage6_observations(
        tracks_file=args.tracks,
        frame_index_path=args.frame_index,
        output_dir=args.output_dir,
        crops_dir=args.crops_dir,
        jpeg_quality=args.quality,
        num_workers=args.workers,
    )
