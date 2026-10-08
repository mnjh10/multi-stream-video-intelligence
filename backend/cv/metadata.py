"""Video metadata extraction module for Person 1 CV pipeline."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import cv2

from backend.cv.discovery import get_repo_root, normalize_repo_path
from backend.cv.validator import validate_video_metadata


def _probe_with_ffprobe(file_path: Path) -> dict[str, Any]:
    """Safely inspect video with ffprobe if available.

    Returns dictionary with any extracted codec, container, or duration info.
    Does not raise exceptions if ffprobe fails or is missing.
    """
    ffprobe_cmd = shutil.which("ffprobe")
    if not ffprobe_cmd:
        return {}

    try:
        cmd = [
            ffprobe_cmd,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-select_streams",
            "v:0",
            "-of",
            "json",
            str(file_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            info: dict[str, Any] = {}
            streams = data.get("streams", [])
            if streams:
                v_stream = streams[0]
                if "codec_name" in v_stream:
                    info["codec"] = v_stream["codec_name"]
            fmt = data.get("format", {})
            if "format_name" in fmt:
                info["container"] = fmt["format_name"]
            if "duration" in fmt:
                try:
                    dur = float(fmt["duration"])
                    if dur > 0 and not math.isnan(dur) and not math.isinf(dur):
                        info["ffprobe_duration"] = round(dur, 3)
                except (ValueError, TypeError):
                    pass
            return info
    except Exception:
        # ffprobe is purely supplementary; never fail metadata extraction if it fails
        pass

    return {}


def extract_video_metadata(
    video_path: Path | str,
    camera_id: str,
    repo_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Extract metadata and validation state from a video file.

    Args:
        video_path: Path to the video file.
        camera_id: Deterministic camera ID (e.g. 'cam_01').
        repo_root: Optional repository root for path normalization.

    Returns:
        Structured metadata dictionary conforming to Person 1 contract.
    """
    root = Path(repo_root).resolve() if repo_root else get_repo_root()
    target_path = Path(video_path)
    if not target_path.is_absolute():
        target_path = root / target_path

    relative_source = normalize_repo_path(target_path, root)

    metadata: dict[str, Any] = {
        "camera_id": camera_id,
        "source_video": relative_source,
        "fps": None,
        "frame_count": None,
        "duration_seconds": None,
        "width": None,
        "height": None,
        "codec": None,
        "container": target_path.suffix.lstrip(".").lower() if target_path.suffix else None,
        "file_size_bytes": None,
        "readable": False,
        "duration_derived": False,
        "validation_status": "invalid",
        "error_message": None,
    }

    # 1. Check existence and file type
    if not target_path.exists():
        metadata["error_message"] = f"Video file not found: {relative_source}"
        return metadata

    if target_path.is_dir():
        from backend.cv.discovery import SUPPORTED_VIDEO_EXTENSIONS

        sub_videos = [
            f
            for f in target_path.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        if len(sub_videos) > 1:
            sub_videos.sort(key=lambda p: (p.name.lower(), p.name))
            candidates = [normalize_repo_path(f, root) for f in sub_videos]
            metadata["candidate_files"] = candidates
            metadata["error_message"] = (
                f"Multiple video files found in camera directory '{target_path.name}': "
                f"{', '.join(candidates)}. Multi-video ambiguity is not permitted; "
                "each camera directory must contain exactly one video file."
            )
            return metadata
        elif len(sub_videos) == 1:
            target_path = sub_videos[0]
            relative_source = normalize_repo_path(target_path, root)
            metadata["source_video"] = relative_source
            metadata["candidate_files"] = [relative_source]
            metadata["container"] = target_path.suffix.lstrip(".").lower() if target_path.suffix else None
        else:
            metadata["error_message"] = f"No supported video files found in directory: {relative_source}"
            return metadata

    try:
        size = target_path.stat().st_size
        metadata["file_size_bytes"] = size
        if size == 0:
            metadata["error_message"] = f"Video file is empty (0 bytes): {relative_source}"
            return metadata
    except OSError as e:
        metadata["error_message"] = f"Cannot read file stat: {e}"
        return metadata

    # 2. Open via OpenCV
    cap = cv2.VideoCapture(str(target_path))
    if not cap.isOpened():
        metadata["error_message"] = "OpenCV failed to open video file (unsupported format or corrupted header)"
        return metadata

    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_val = float(cap.get(cv2.CAP_PROP_FPS))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        metadata["width"] = w if w > 0 else None
        metadata["height"] = h if h > 0 else None
        metadata["fps"] = round(fps_val, 3) if fps_val > 0 and not math.isnan(fps_val) and not math.isinf(fps_val) else None
        metadata["frame_count"] = fc if fc > 0 else None

        # Decode fourcc if available
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc_int != 0:
            fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            if fourcc_str.isprintable() and len(fourcc_str.strip()) > 0:
                metadata["codec"] = fourcc_str.strip()

        # 3. Test readability by decoding the first frame
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            metadata["readable"] = True
        else:
            metadata["readable"] = False
            metadata["error_message"] = "Failed to decode the first frame from the video stream"

    except Exception as e:
        metadata["readable"] = False
        metadata["error_message"] = f"Exception while reading video properties: {e}"
    finally:
        cap.release()

    # 4. Determine duration
    if metadata["fps"] and metadata["frame_count"] and metadata["fps"] > 0 and metadata["frame_count"] > 0:
        metadata["duration_seconds"] = round(metadata["frame_count"] / metadata["fps"], 3)
        metadata["duration_derived"] = True

    # 5. Enrich with ffprobe if available
    ff_info = _probe_with_ffprobe(target_path)
    if ff_info:
        if ff_info.get("codec"):
            metadata["codec"] = ff_info["codec"]
        if ff_info.get("container"):
            metadata["container"] = ff_info["container"]
        # If ffprobe reports duration and OpenCV could not derive one, use ffprobe duration
        if metadata["duration_seconds"] is None and ff_info.get("ffprobe_duration"):
            metadata["duration_seconds"] = ff_info["ffprobe_duration"]
            metadata["duration_derived"] = False

    # 6. Validate metadata against pipeline invariants
    is_valid, validation_err = validate_video_metadata(metadata)
    if is_valid:
        metadata["validation_status"] = "valid"
        metadata["error_message"] = None
    else:
        metadata["validation_status"] = "invalid"
        if not metadata["error_message"]:
            metadata["error_message"] = validation_err

    return metadata
