"""Person 1 Stage 6: Object Crop Extraction and Validation.

Extracts deterministic bounding-box pixel regions from Stage 3 sampled frames
and serializes them as individual JPEG crops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("argus.cv.crops")

DEFAULT_CROP_QUALITY: int = 95


@dataclass
class CropConfig:
    """Configuration for object crop extraction."""

    jpeg_quality: int = DEFAULT_CROP_QUALITY
    crops_dir: str = "data/crops"

    def __post_init__(self) -> None:
        """Validate crop configuration."""
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality must be in [1, 100], got {self.jpeg_quality}")


def compute_safe_crop_coordinates(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Convert float bbox [x1, y1, x2, y2] to safe integer pixel coordinates.

    Ensures coordinates are bounded within [0, image_width] and [0, image_height],
    and that width and height are strictly >= 1 pixel.

    Args:
        bbox: Float bounding box [x1, y1, x2, y2].
        image_width: Native image width in pixels.
        image_height: Native image height in pixels.

    Returns:
        Tuple of (x1_int, y1_int, x2_int, y2_int).
    """
    x1, y1, x2, y2 = bbox

    x1_int = max(0, min(int(round(x1)), image_width - 1))
    y1_int = max(0, min(int(round(y1)), image_height - 1))
    x2_int = max(x1_int + 1, min(int(round(x2)), image_width))
    y2_int = max(y1_int + 1, min(int(round(y2)), image_height))

    return x1_int, y1_int, x2_int, y2_int


def extract_and_save_crop(
    frame_image: np.ndarray,
    bbox: list[float],
    output_path: Path | str,
    jpeg_quality: int = DEFAULT_CROP_QUALITY,
) -> tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
    """Extract bounding box region from a decoded frame and write to disk as JPEG.

    Args:
        frame_image: Decoded numpy frame array (H, W, C).
        bbox: Bounding box [x1, y1, x2, y2] in native resolution.
        output_path: Target JPEG file path.
        jpeg_quality: JPEG compression quality setting [1, 100].

    Returns:
        Tuple of (success, (crop_width, crop_height), error_reason).
    """
    if frame_image is None or frame_image.size == 0:
        return False, None, "Invalid or empty source frame image"

    img_h, img_w = frame_image.shape[:2]
    if img_w <= 0 or img_h <= 0:
        return False, None, f"Invalid frame dimensions ({img_w}x{img_h})"

    x1, y1, x2, y2 = compute_safe_crop_coordinates(bbox, img_w, img_h)
    crop = frame_image[y1:y2, x1:x2]

    crop_h, crop_w = crop.shape[:2]
    if crop_w <= 0 or crop_h <= 0:
        return False, None, f"Extracted crop has zero area ({crop_w}x{crop_h})"

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    success = cv2.imwrite(str(out_p), crop, encode_params)
    if not success or not out_p.exists():
        return False, None, f"Failed to write crop image to {out_p}"

    return True, (crop_w, crop_h), None


def validate_crop_file(crop_path: Path | str) -> tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
    """Validate that a generated crop exists on disk and is readable with valid dimensions.

    Args:
        crop_path: Path to the crop JPEG file.

    Returns:
        Tuple of (is_valid, (width, height), error_reason).
    """
    p = Path(crop_path)
    if not p.exists():
        return False, None, f"Crop file does not exist: {p}"

    if p.stat().st_size == 0:
        return False, None, f"Crop file is empty (0 bytes): {p}"

    img = cv2.imread(str(p))
    if img is None or img.size == 0:
        return False, None, f"Failed to decode crop image: {p}"

    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return False, None, f"Crop has invalid dimensions ({w}x{h}): {p}"

    return True, (w, h), None
