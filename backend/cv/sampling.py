"""Deterministic frame sampling and timestamp indexing for Person 1 CV pipeline.

Extracts frames at a configurable sampling rate (default 3.0 FPS) from source
CCTV videos, preserving native resolution, applying deterministic 0-indexed
frame positions, and computing source-video elapsed timestamps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2

from backend.cv.discovery import get_repo_root, normalize_repo_path

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_FPS: float = 3.0
DEFAULT_JPEG_QUALITY: int = 95
TIMESTAMP_PRECISION: int = 4


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration parameters for deterministic frame sampling."""

    sample_fps: float = DEFAULT_SAMPLE_FPS
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    output_dir: str = "data/frames"
    index_file: str = "data/frames/frame_index.jsonl"
    manifest_file: str = "data/frames/sampling_manifest.json"

    def __post_init__(self):
        if self.sample_fps <= 0:
            raise ValueError(f"sample_fps must be positive, got {self.sample_fps}")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality must be in [1, 100], got {self.jpeg_quality}")


def compute_sample_indices(
    frame_count: int,
    source_fps: float,
    sample_fps: float,
) -> list[int]:
    """Derive deterministic 0-indexed source frame indices for a sampling schedule.

    For a video with source framerate F and desired sample framerate S:
    Desired sample times: t_k = k / S
    Corresponding source frame index: round(k * F / S)

    If S >= F, all source frames [0, frame_count - 1] are selected without duplication.

    Args:
        frame_count: Total number of frames in the source video.
        source_fps: Actual native framerate of the source video.
        sample_fps: Desired sampling rate in frames per second.

    Returns:
        List of unique, strictly increasing 0-based source frame indices.
    """
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be strictly positive, got {sample_fps}")
    if source_fps <= 0:
        raise ValueError(f"source_fps must be strictly positive, got {source_fps}")
    if frame_count <= 0:
        return []

    # If sampling rate meets or exceeds native capture, sample all available frames
    if sample_fps >= source_fps:
        return list(range(frame_count))

    indices: list[int] = []
    step = source_fps / sample_fps
    k = 0
    while True:
        idx = int(round(k * step))
        if idx >= frame_count:
            break
        indices.append(idx)
        k += 1

    # Guarantee uniqueness and deterministic ordering
    unique_indices = sorted(list(set(indices)))
    return unique_indices


def compute_timestamp(
    frame_index: int,
    source_fps: float,
    precision: Optional[int] = TIMESTAMP_PRECISION,
) -> float:
    """Compute elapsed source-video seconds for a given frame index.

    Formula:
        timestamp = frame_index / source_video_fps

    Per the Stage 0 contract, synchronization offsets (e.g. CityFlowV2 offset_seconds)
    are NOT added to this observation timestamp.

    Args:
        frame_index: 0-indexed position within the source video.
        source_fps: Native framerate of the source video.
        precision: Decimal rounding precision. If None, returns unrounded float.
                   Default is 4 (TIMESTAMP_PRECISION), which formats and stores
                   clean terminating decimals (e.g. 0.3, 0.375) without IEEE 754 residue.

    Returns:
        Elapsed source-video time in seconds as a float.
    """
    if source_fps <= 0:
        raise ValueError(f"source_fps must be strictly positive, got {source_fps}")
    if frame_index < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame_index}")

    raw_ts = float(frame_index) / float(source_fps)
    return round(raw_ts, precision) if precision is not None else raw_ts



def sample_camera_video(
    camera_record: dict[str, Any],
    config: SamplingConfig,
    repo_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Sample frames deterministically from a single camera's source video.

    Args:
        camera_record: Metadata dictionary for the camera (from inventory/manifest).
        config: Sampling configuration options.
        repo_root: Repository root path for relative path resolution.

    Returns:
        Dictionary containing camera sampling results, statistics, and frame records.
    """
    root = Path(repo_root).resolve() if repo_root else get_repo_root()

    camera_id = camera_record["camera_id"]
    source_rel = camera_record.get("source_video")
    if not source_rel:
        raise ValueError(f"Camera '{camera_id}' has no source_video specified")

    source_path = root / source_rel if not Path(source_rel).is_absolute() else Path(source_rel)
    if not source_path.exists():
        raise FileNotFoundError(f"Source video file not found for camera '{camera_id}': {source_path}")

    # Validate FPS
    source_fps = camera_record.get("fps")
    if source_fps is None or source_fps <= 0:
        raise ValueError(f"Camera '{camera_id}' has invalid source FPS: {source_fps}")

    # Validate frame count
    frame_count = camera_record.get("frame_count")
    if frame_count is None or frame_count <= 0:
        raise ValueError(f"Camera '{camera_id}' has invalid frame count: {frame_count}")

    # Expected dimensions
    expected_width = camera_record.get("width")
    expected_height = camera_record.get("height")

    # Output directory for camera frames
    cam_frames_dir = root / config.output_dir / camera_id
    cam_frames_dir.mkdir(parents=True, exist_ok=True)

    # Compute planned sample indices
    target_indices = compute_sample_indices(
        frame_count=frame_count,
        source_fps=source_fps,
        sample_fps=config.sample_fps,
    )
    target_set = set(target_indices)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video file for camera '{camera_id}': {source_path}")

    frame_records: list[dict[str, Any]] = []
    current_idx = 0
    saved_count = 0
    t0 = time.time()

    try:
        while current_idx < frame_count:
            ret = cap.grab()
            if not ret:
                # If grab fails before reaching expected frame_count, record explicit failure
                raise RuntimeError(
                    f"Camera '{camera_id}': Failed to grab frame {current_idx} "
                    f"(expected {frame_count} total frames) from {source_rel}"
                )

            if current_idx in target_set:
                ret_decode, frame = cap.retrieve()
                if not ret_decode or frame is None:
                    raise RuntimeError(
                        f"Camera '{camera_id}': Failed to decode/retrieve frame {current_idx} from {source_rel}"
                    )

                h, w = frame.shape[:2]
                if expected_width is not None and expected_height is not None:
                    if w != expected_width or h != expected_height:
                        raise ValueError(
                            f"Camera '{camera_id}': Frame {current_idx} dimensions ({w}x{h}) "
                            f"do not match expected native dimensions ({expected_width}x{expected_height})"
                        )

                # Deterministic filename: frame_{current_idx:06d}.jpg
                frame_filename = f"frame_{current_idx:06d}.jpg"
                frame_abs_path = cam_frames_dir / frame_filename
                frame_rel_path = f"{config.output_dir}/{camera_id}/{frame_filename}".replace("\\", "/")

                # Write JPEG without resizing
                write_success = cv2.imwrite(
                    str(frame_abs_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
                )
                if not write_success or not frame_abs_path.exists() or frame_abs_path.stat().st_size == 0:
                    raise IOError(
                        f"Camera '{camera_id}': Failed to write sampled frame {current_idx} to {frame_abs_path}"
                    )

                record = {
                    "camera_id": camera_id,
                    "frame_index": current_idx,
                    "timestamp": compute_timestamp(current_idx, source_fps),
                    "frame_path": frame_rel_path,
                    "source_video": normalize_repo_path(source_path, root),
                }
                frame_records.append(record)
                saved_count += 1

            current_idx += 1
    finally:
        cap.release()

    elapsed = time.time() - t0
    disk_bytes = sum((cam_frames_dir / f"frame_{r['frame_index']:06d}.jpg").stat().st_size for r in frame_records)

    return {
        "camera_id": camera_id,
        "source_video": normalize_repo_path(source_path, root),
        "source_fps": source_fps,
        "total_source_frames": frame_count,
        "sampled_frames_count": saved_count,
        "planned_samples_count": len(target_indices),
        "native_width": expected_width or w,
        "native_height": expected_height or h,
        "processing_time_seconds": round(elapsed, 3),
        "frames_disk_bytes": disk_bytes,
        "frame_records": frame_records,
    }


def run_stage3_sampling(
    inventory_path: Path | str = "data/observations/camera_inventory.json",
    video_dir: Path | str = "data/videos",
    output_dir: Path | str = "data/frames",
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    repo_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Execute the full Stage 3 frame sampling pipeline across all cameras.

    Args:
        inventory_path: Path to camera inventory manifest JSON.
        video_dir: Directory containing camera video feeds (fallback if inventory absent).
        output_dir: Target root directory for sampled frames.
        sample_fps: Configurable sampling framerate target.
        jpeg_quality: JPEG compression quality factor (1-100).
        repo_root: Repository root path for relative paths.

    Returns:
        Structured dictionary summarizing sampling execution across all cameras.
    """
    root = Path(repo_root).resolve() if repo_root else get_repo_root()
    config = SamplingConfig(
        sample_fps=sample_fps,
        jpeg_quality=jpeg_quality,
        output_dir=str(Path(output_dir).as_posix()),
        index_file=f"{output_dir}/frame_index.jsonl".replace("\\", "/"),
        manifest_file=f"{output_dir}/sampling_manifest.json".replace("\\", "/"),
    )

    inv_path = root / inventory_path if not Path(inventory_path).is_absolute() else Path(inventory_path)
    if not inv_path.exists():
        from backend.cv.inventory import build_camera_inventory
        logger.info(f"Inventory {inv_path} not found; building dynamically from {video_dir}...")
        inventory = build_camera_inventory(video_dir=video_dir, repo_root=root)
    else:
        with open(inv_path, "r", encoding="utf-8") as f:
            inventory = json.load(f)

    cameras = inventory.get("cameras", [])
    valid_cameras = [c for c in cameras if c.get("validation_status") == "valid"]
    if not valid_cameras:
        raise RuntimeError("No valid cameras found in inventory for frame sampling")

    # Sort cameras deterministically by camera_id
    valid_cameras = sorted(valid_cameras, key=lambda c: (c["camera_id"].lower(), c["camera_id"]))

    all_frame_records: list[dict[str, Any]] = []
    camera_results: list[dict[str, Any]] = []
    total_start = time.time()

    print("=" * 65)
    print("STAGE 3 — DETERMINISTIC FRAME SAMPLING & TIMESTAMP INDEXING")
    print(f"Sample target rate: {config.sample_fps} FPS | JPEG Quality: {config.jpeg_quality}")
    print(f"Processing {len(valid_cameras)} valid cameras...")
    print("=" * 65)

    for cam in valid_cameras:
        cam_id = cam["camera_id"]
        source_fps = cam["fps"]
        print(f"[{cam_id}] Sampling {cam['source_video']} ({source_fps} FPS, {cam['frame_count']} frames)...")

        res = sample_camera_video(camera_record=cam, config=config, repo_root=root)
        camera_results.append(res)
        all_frame_records.extend(res["frame_records"])

        fps_speed = (
            round(res["total_source_frames"] / res["processing_time_seconds"], 1)
            if res["processing_time_seconds"] > 0
            else 0
        )
        mb_disk = round(res["frames_disk_bytes"] / (1024 * 1024), 2)
        print(
            f"  -> Extracted {res['sampled_frames_count']} frames in {res['processing_time_seconds']}s "
            f"({fps_speed} source fps, {mb_disk} MB)"
        )

    total_elapsed = time.time() - total_start
    total_source_frames = sum(r["total_source_frames"] for r in camera_results)
    total_sampled_frames = len(all_frame_records)
    total_disk_bytes = sum(r["frames_disk_bytes"] for r in camera_results)

    # 1. Write frame_index.jsonl (Deterministic line-delimited index)
    index_path = root / config.index_file
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        for rec in all_frame_records:
            line = json.dumps(rec, ensure_ascii=False)
            f.write(line + "\n")

    # 2. Write sampling_manifest.json (Deterministic summary manifest)
    cam_summaries = []
    for r in camera_results:
        cam_summaries.append({
            "camera_id": r["camera_id"],
            "source_video": r["source_video"],
            "source_fps": r["source_fps"],
            "native_width": r["native_width"],
            "native_height": r["native_height"],
            "total_source_frames": r["total_source_frames"],
            "sampled_frames_count": r["sampled_frames_count"],
            "frames_disk_bytes": r["frames_disk_bytes"],
            "first_sampled_frame": r["frame_records"][0]["frame_index"] if r["frame_records"] else None,
            "last_sampled_frame": r["frame_records"][-1]["frame_index"] if r["frame_records"] else None,
        })

    manifest = {
        "schema_version": "1.0",
        "sample_fps": config.sample_fps,
        "jpeg_quality": config.jpeg_quality,
        "output_directory": normalize_repo_path(root / config.output_dir, root),
        "frame_index_file": normalize_repo_path(index_path, root),
        "total_cameras": len(camera_results),
        "total_source_frames": total_source_frames,
        "total_sampled_frames": total_sampled_frames,
        "total_frames_disk_bytes": total_disk_bytes,
        "cameras": cam_summaries,
    }

    manifest_path = root / config.manifest_file
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print(f"SAMPLING COMPLETE: {total_sampled_frames} frames extracted from {total_source_frames} source frames.")
    print(f"Total processing time: {total_elapsed:.2f}s (avg {(total_source_frames / total_elapsed):.1f} source fps)")
    print(f"Total disk usage: {total_disk_bytes / (1024 * 1024):.2f} MB")
    print(f"Frame index written to: {normalize_repo_path(index_path, root)}")
    print(f"Sampling manifest written to: {normalize_repo_path(manifest_path, root)}")
    print("=" * 65)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person 1 Deterministic Frame Sampling")
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS, help="Sampling framerate target")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality factor (1-100)")
    parser.add_argument("--inventory", default="data/observations/camera_inventory.json", help="Path to camera inventory JSON")
    parser.add_argument("--video-dir", default="data/videos", help="Directory containing camera videos")
    parser.add_argument("--output-dir", default="data/frames", help="Output directory for sampled frames")
    args = parser.parse_args()

    run_stage3_sampling(
        inventory_path=args.inventory,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        sample_fps=args.sample_fps,
        jpeg_quality=args.jpeg_quality,
    )
