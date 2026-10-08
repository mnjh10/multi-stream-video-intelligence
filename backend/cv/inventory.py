"""Camera inventory builder and manifest serializer for Person 1 CV pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.cv.discovery import (
    assign_camera_ids,
    discover_video_sources,
    get_repo_root,
    normalize_repo_path,
)
from backend.cv.metadata import extract_video_metadata

logger = logging.getLogger(__name__)


def build_camera_inventory(
    video_dir: Path | str = "data/videos",
    repo_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Discover, inspect, validate, and record all camera video sources.

    Args:
        video_dir: Directory containing input camera feeds or folders.
        repo_root: Optional root directory of the repository.

    Returns:
        Structured camera inventory dictionary.
    """
    root = Path(repo_root).resolve() if repo_root else get_repo_root()
    v_dir = Path(video_dir)
    if not v_dir.is_absolute():
        v_dir = root / v_dir

    # 1. Discover video sources deterministically
    discovered_sources = discover_video_sources(video_dir=v_dir, repo_root=root)

    # 2. Assign canonical, collision-free camera IDs
    assigned_sources = assign_camera_ids(discovered_sources)

    # 3. Extract metadata and validate each video
    cameras: list[dict[str, Any]] = []
    for src in assigned_sources:
        cam_id = src["camera_id"]
        source_path = src["source_path"]
        if src.get("is_ambiguous"):
            meta = {
                "camera_id": cam_id,
                "source_video": src["relative_path"],
                "fps": None,
                "frame_count": None,
                "duration_seconds": None,
                "width": None,
                "height": None,
                "codec": None,
                "container": None,
                "file_size_bytes": None,
                "readable": False,
                "duration_derived": False,
                "validation_status": "invalid",
                "error_message": src.get("ambiguity_error"),
                "candidate_files": src.get("candidate_files", []),
            }
        else:
            meta = extract_video_metadata(
                video_path=source_path,
                camera_id=cam_id,
                repo_root=root,
            )
            if "candidate_files" in src:
                meta["candidate_files"] = src["candidate_files"]
        cameras.append(meta)

    valid_count = sum(1 for c in cameras if c.get("validation_status") == "valid")
    invalid_count = len(cameras) - valid_count

    inventory: dict[str, Any] = {
        "schema_version": "1.0",
        "video_directory": normalize_repo_path(v_dir, root),
        "total_cameras": len(cameras),
        "valid_cameras": valid_count,
        "invalid_cameras": invalid_count,
        "cameras": cameras,
    }

    return inventory


def save_camera_inventory(
    inventory: dict[str, Any],
    output_path: Path | str = "data/observations/camera_inventory.json",
    repo_root: Optional[Path | str] = None,
) -> Path:
    """Serialize the camera inventory to disk in JSON format.

    Args:
        inventory: The camera inventory dictionary.
        output_path: Target JSON file path.
        repo_root: Optional repository root for path resolution.

    Returns:
        The resolved Path where the manifest was written.
    """
    root = Path(repo_root).resolve() if repo_root else get_repo_root()
    out = Path(output_path)
    if not out.is_absolute():
        out = root / out

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)

    return out


def run_stage1_pipeline(
    video_dir: str = "data/videos",
    output_path: str = "data/observations/camera_inventory.json",
    repo_root: Optional[str] = None,
) -> dict[str, Any]:
    """Execute Stage 1: build camera inventory and save manifest."""
    root = Path(repo_root).resolve() if repo_root else get_repo_root()
    inventory = build_camera_inventory(video_dir=video_dir, repo_root=root)
    saved_path = save_camera_inventory(inventory, output_path=output_path, repo_root=root)

    print("=" * 60)
    print("STAGE 1 - CAMERA INVENTORY RUN")
    print(f"Video directory: {inventory['video_directory']}")
    print(f"Total cameras discovered: {inventory['total_cameras']}")
    print(f"Valid cameras: {inventory['valid_cameras']}")
    print(f"Invalid cameras: {inventory['invalid_cameras']}")
    print(f"Manifest saved to: {normalize_repo_path(saved_path, root)}")
    if inventory["total_cameras"] == 0:
        print("NO REAL INPUT VIDEOS DISCOVERED IN data/videos")
    print("=" * 60)

    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person 1 Video Ingestion & Camera Inventory")
    parser.add_argument("--video-dir", default="data/videos", help="Directory containing input videos")
    parser.add_argument("--output", default="data/observations/camera_inventory.json", help="Manifest output JSON path")
    args = parser.parse_args()

    run_stage1_pipeline(video_dir=args.video_dir, output_path=args.output)
