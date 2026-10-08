"""Video discovery and deterministic camera ID assignment for Person 1."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Set

SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
}

CANONICAL_CAM_PATTERN = re.compile(r"^cam[_-]?(\d+)$", re.IGNORECASE)


def get_repo_root() -> Path:
    """Return the absolute path of the repository root."""
    # Assuming backend/cv/discovery.py -> repo_root is 2 levels up from backend
    current_file = Path(__file__).resolve()
    # current_file: .../backend/cv/discovery.py -> parent: cv -> parent: backend -> parent: repo root
    return current_file.parent.parent.parent


def normalize_repo_path(path: Path | str, repo_root: Optional[Path | str] = None) -> str:
    """Convert a path into a POSIX repository-relative path."""
    target = Path(path).resolve()
    root = Path(repo_root).resolve() if repo_root else get_repo_root()

    try:
        rel = target.relative_to(root)
        return rel.as_posix()
    except ValueError:
        # If not relative to repo root, return normalized relative path or POSIX representation
        return target.as_posix()


def discover_video_sources(
    video_dir: Path | str = "data/videos",
    repo_root: Optional[Path | str] = None,
) -> List[dict[str, Any]]:
    """Discover video sources deterministically under the video directory.

    Supported directory conventions:
    1. Subdirectory layout: data/videos/<camera_name>/<video_file>
       - Exactly one supported video file: processed normally.
       - More than one supported video file: flagged as ambiguous with all
         candidate files reported. No file is silently discarded.
    2. Direct file layout: data/videos/<video_file>
       Direct video files in the root video directory are treated as independent streams.

    Returns:
        List of dicts sorted deterministically by repository-relative POSIX path.
    """
    root = Path(repo_root).resolve() if repo_root else get_repo_root()
    base_dir = Path(video_dir)
    if not base_dir.is_absolute():
        base_dir = root / base_dir

    if not base_dir.exists() or not base_dir.is_dir():
        return []

    discovered: List[dict[str, Any]] = []

    # Iterate over entries in sorted order to maintain determinism
    for entry in sorted(base_dir.iterdir(), key=lambda p: (p.name.lower(), p.name)):
        # Ignore hidden files/directories (e.g., .git, .DS_Store)
        if entry.name.startswith("."):
            continue

        if entry.is_dir():
            # Check for video files in this camera subdirectory
            video_files = [
                f
                for f in entry.iterdir()
                if f.is_file()
                and not f.name.startswith(".")
                and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
            ]
            if len(video_files) == 1:
                chosen = video_files[0]
                rel_path = normalize_repo_path(chosen, root)
                discovered.append(
                    {
                        "source_path": chosen.resolve(),
                        "relative_path": rel_path,
                        "source_name": entry.name,
                        "is_subfolder": True,
                        "candidate_files": [rel_path],
                        "is_ambiguous": False,
                        "ambiguity_error": None,
                    }
                )
            elif len(video_files) > 1:
                # Ambiguity: more than one video file in the same camera directory
                video_files.sort(key=lambda p: (p.name.lower(), p.name))
                candidates = [normalize_repo_path(f, root) for f in video_files]
                dir_rel_path = normalize_repo_path(entry, root)
                discovered.append(
                    {
                        "source_path": entry.resolve(),
                        "relative_path": dir_rel_path,
                        "source_name": entry.name,
                        "is_subfolder": True,
                        "candidate_files": candidates,
                        "is_ambiguous": True,
                        "ambiguity_error": (
                            f"Multiple video files found in camera directory '{entry.name}': "
                            f"{', '.join(candidates)}. Multi-video ambiguity is not permitted; "
                            "each camera directory must contain exactly one video file."
                        ),
                    }
                )
        elif entry.is_file() and entry.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            file_rel_path = normalize_repo_path(entry, root)
            discovered.append(
                {
                    "source_path": entry.resolve(),
                    "relative_path": file_rel_path,
                    "source_name": entry.stem,
                    "is_subfolder": False,
                    "candidate_files": [file_rel_path],
                    "is_ambiguous": False,
                    "ambiguity_error": None,
                }
            )

    # Global deterministic sort across all discovered sources
    discovered.sort(key=lambda d: (d["relative_path"].lower(), d["relative_path"]))
    return discovered


def assign_camera_ids(discovered_sources: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Assign deterministic canonical camera IDs (cam_01, cam_02, ...) to discovered sources.

    Rules:
    1. If the source name (subfolder name or file stem) matches canonical patterns
       such as 'cam_01', 'cam1', 'CAM-02', normalize it directly to 'cam_XX'.
    2. Any unmatched sources are assigned the next available canonical sequential ID
       'cam_XX' (1-indexed) based on deterministic sorted order.
    3. Guarantees no ID collisions.

    Returns:
        List of source dicts with 'camera_id' field added.
    """
    assigned: List[dict[str, Any]] = []
    used_ids: Set[str] = set()
    unassigned_entries: List[dict[str, Any]] = []

    # First pass: preserve and normalize explicit canonical names
    for entry in discovered_sources:
        entry_copy = dict(entry)
        name = entry_copy["source_name"]
        match = CANONICAL_CAM_PATTERN.match(name)
        if match:
            num = int(match.group(1))
            canonical_id = f"cam_{num:02d}"
            if canonical_id not in used_ids:
                entry_copy["camera_id"] = canonical_id
                used_ids.add(canonical_id)
                assigned.append(entry_copy)
                continue
        # Mark for sequential assignment
        unassigned_entries.append(entry_copy)

    # Second pass: assign sequential IDs to remaining entries
    counter = 1
    for entry in unassigned_entries:
        while f"cam_{counter:02d}" in used_ids:
            counter += 1
        canonical_id = f"cam_{counter:02d}"
        used_ids.add(canonical_id)
        entry["camera_id"] = canonical_id
        assigned.append(entry)

    # Sort final list by camera_id deterministically
    assigned.sort(key=lambda d: d["camera_id"])
    return assigned
