"""
Visual Attribute and Color Verifier for ARGUS.
Performs lightweight, deterministic verification of visual attributes (e.g. vehicle color)
directly on Person 1 object crops using calibrated HSV distribution thresholds.
"""

from pathlib import Path
from typing import Any
import cv2
import numpy as np


class AttributeVerifier:
    """
    Verifies visual attributes (colors) on object crops using empirical HSV thresholds.
    Ensures that retrieval candidates are not falsely presented as matching attributes
    merely due to high generic semantic similarity scores.
    """

    # Calibrated OpenCV HSV bounds: H in [0, 180], S in [0, 255], V in [0, 255]
    COLOR_RANGES = {
        "red": [
            (np.array([0, 70, 50]), np.array([10, 255, 255])),
            (np.array([170, 70, 50]), np.array([180, 255, 255])),
        ],
        "blue": [
            (np.array([100, 70, 50]), np.array([135, 255, 255])),
        ],
        "green": [
            (np.array([35, 60, 50]), np.array([85, 255, 255])),
        ],
        "yellow": [
            (np.array([15, 80, 70]), np.array([35, 255, 255])),
        ],
        "white": [
            # Low saturation, high brightness
            (np.array([0, 0, 160]), np.array([180, 45, 255])),
        ],
        "black": [
            # Very low brightness
            (np.array([0, 0, 0]), np.array([180, 255, 55])),
        ],
        "purple": [
            (np.array([125, 50, 50]), np.array([160, 255, 255])),
        ],
        "orange": [
            (np.array([10, 80, 70]), np.array([25, 255, 255])),
        ],
        "gray": [
            # Low saturation, medium brightness
            (np.array([0, 0, 55]), np.array([180, 40, 160])),
        ],
        "grey": [
            (np.array([0, 0, 55]), np.array([180, 40, 160])),
        ],
        "silver": [
            (np.array([0, 0, 120]), np.array([180, 30, 220])),
        ],
    }

    # Conservative acceptance thresholds established through empirical inspection of CityFlow crops
    THRESHOLDS = {
        "red": 0.18,
        "blue": 0.18,
        "green": 0.18,
        "yellow": 0.18,
        "purple": 0.15,
        "orange": 0.18,
        "white": 0.25,
        "black": 0.25,
        "gray": 0.25,
        "grey": 0.25,
        "silver": 0.25,
    }

    @classmethod
    def supported_attributes(cls) -> set[str]:
        return set(cls.COLOR_RANGES.keys())

    def verify_crop(self, crop_path: str | Path | None, attribute: str) -> dict[str, Any]:
        """
        Verify if the image crop at crop_path matches the specified attribute.

        Returns a dictionary with:
            - verified (bool): Whether the attribute exceeds conservative acceptance threshold.
            - status (str): 'attribute_verified', 'unverified_candidate', 'rejected', or 'unsupported_attribute'.
            - fraction (float): Fraction of pixels in vehicle ROI matching the attribute.
            - threshold (float): Target acceptance threshold.
            - attribute (str): The queried attribute.
        """
        attr_norm = attribute.strip().lower()

        if attr_norm not in self.COLOR_RANGES:
            return {
                "verified": False,
                "status": "unsupported_attribute",
                "fraction": 0.0,
                "threshold": 0.0,
                "attribute": attr_norm,
                "details": f"Attribute '{attr_norm}' is not supported for direct color verification.",
            }

        if not crop_path:
            return {
                "verified": False,
                "status": "unverified_candidate",
                "fraction": 0.0,
                "threshold": self.THRESHOLDS[attr_norm],
                "attribute": attr_norm,
                "details": "Crop path is missing.",
            }

        crop_file = Path(crop_path)
        if not crop_file.exists():
            return {
                "verified": False,
                "status": "unverified_candidate",
                "fraction": 0.0,
                "threshold": self.THRESHOLDS[attr_norm],
                "attribute": attr_norm,
                "details": f"Crop file not found: {crop_path}",
            }

        img = cv2.imread(str(crop_file))
        if img is None or img.size == 0:
            return {
                "verified": False,
                "status": "unverified_candidate",
                "fraction": 0.0,
                "threshold": self.THRESHOLDS[attr_norm],
                "attribute": attr_norm,
                "details": "Failed to decode crop image.",
            }

        h, w = img.shape[:2]
        # Crop inner 80% to focus on vehicle body and minimize road asphalt border
        y_margin = int(h * 0.1)
        x_margin = int(w * 0.1)
        roi = img[y_margin : max(y_margin + 1, h - y_margin), x_margin : max(x_margin + 1, w - x_margin)]
        if roi.size == 0:
            roi = img

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = roi.shape[0] * roi.shape[1]

        ranges = self.COLOR_RANGES[attr_norm]
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        fraction = float(np.count_nonzero(mask) / max(1, total_pixels))
        threshold = self.THRESHOLDS[attr_norm]

        # For chromatic colors (especially red), verify that the vehicle is not predominantly white/gray
        # (which happens when brake lights illuminate on white/silver cars)
        if attr_norm in ("red", "blue", "green", "yellow"):
            white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 45, 255]))
            white_fraction = float(np.count_nonzero(white_mask) / max(1, total_pixels))
            # If the vehicle has more white than the target chromatic color and white > 25%,
            # it is predominantly a white vehicle with localized chromatic reflections (e.g. tail lights)
            if white_fraction > 0.25 and white_fraction > fraction:
                return {
                    "verified": False,
                    "status": "rejected",
                    "fraction": round(fraction, 4),
                    "threshold": threshold,
                    "attribute": attr_norm,
                    "details": f"Rejected: white body dominance ({white_fraction*100:.1f}%) exceeds {attr_norm} ({fraction*100:.1f}%).",
                }

        if fraction >= threshold:
            status = "attribute_verified"
            verified = True
        elif fraction >= (threshold * 0.55):
            status = "unverified_candidate"
            verified = False
        else:
            status = "rejected"
            verified = False

        return {
            "verified": verified,
            "status": status,
            "fraction": round(fraction, 4),
            "threshold": threshold,
            "attribute": attr_norm,
            "details": f"{attr_norm.capitalize()} pixel fraction {fraction*100:.1f}% vs threshold {threshold*100:.1f}%.",
        }
