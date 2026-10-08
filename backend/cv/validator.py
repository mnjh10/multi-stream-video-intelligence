"""Video metadata validation for Person 1 CV pipeline."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple


def validate_video_metadata(metadata: dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate video metadata dictionary against required CV pipeline invariants.

    Requirements for a VALID video:
    - File exists and has positive file size
    - Video stream is readable (first frame decodes successfully)
    - Positive width and height dimensions
    - Positive, finite FPS (> 0)
    - Positive frame count (> 0)
    - Positive, finite duration in seconds (> 0)

    Returns:
        Tuple of (is_valid: bool, error_message: str | None).
    """
    file_size = metadata.get("file_size_bytes")
    if file_size is None or file_size <= 0:
        return False, "Video file is missing, unreadable, or empty (0 bytes)"

    if not metadata.get("readable", False):
        error = metadata.get("error_message") or "Video cannot be decoded or opened by OpenCV"
        return False, error

    width = metadata.get("width")
    height = metadata.get("height")
    if not isinstance(width, (int, float)) or width <= 0 or not isinstance(height, (int, float)) or height <= 0:
        return False, f"Invalid video dimensions: width={width}, height={height}"

    fps = metadata.get("fps")
    if fps is None or not isinstance(fps, (int, float)) or fps <= 0 or math.isnan(fps) or math.isinf(fps):
        return False, f"Invalid or non-positive FPS: {fps}"

    frame_count = metadata.get("frame_count")
    if frame_count is None or not isinstance(frame_count, (int, float)) or frame_count <= 0:
        return False, f"Invalid or non-positive frame count: {frame_count}"

    duration = metadata.get("duration_seconds")
    if duration is None or not isinstance(duration, (int, float)) or duration <= 0 or math.isnan(duration) or math.isinf(duration):
        return False, f"Invalid or non-positive duration: {duration}"

    return True, None
