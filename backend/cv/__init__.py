"""Person 1 Computer Vision & Ingestion Pipeline."""

from backend.cv.discovery import (
    assign_camera_ids,
    discover_video_sources,
    normalize_repo_path,
)
from backend.cv.metadata import extract_video_metadata
from backend.cv.validator import validate_video_metadata


def build_camera_inventory(*args, **kwargs):
    from backend.cv.inventory import build_camera_inventory as _build
    return _build(*args, **kwargs)


def save_camera_inventory(*args, **kwargs):
    from backend.cv.inventory import save_camera_inventory as _save
    return _save(*args, **kwargs)


def run_stage1_pipeline(*args, **kwargs):
    from backend.cv.inventory import run_stage1_pipeline as _run
    return _run(*args, **kwargs)


def compute_sample_indices(*args, **kwargs):
    from backend.cv.sampling import compute_sample_indices as _csi
    return _csi(*args, **kwargs)


def compute_timestamp(*args, **kwargs):
    from backend.cv.sampling import compute_timestamp as _ct
    return _ct(*args, **kwargs)


def run_stage3_sampling(*args, **kwargs):
    from backend.cv.sampling import run_stage3_sampling as _s3
    return _s3(*args, **kwargs)


def validate_and_clip_bbox(*args, **kwargs):
    from backend.cv.detection import validate_and_clip_bbox as _vacb
    return _vacb(*args, **kwargs)


def run_stage4_detection(*args, **kwargs):
    from backend.cv.detection import run_stage4_detection as _s4
    return _s4(*args, **kwargs)


def validate_track_record(*args, **kwargs):
    from backend.cv.tracking import validate_track_record as _vtr
    return _vtr(*args, **kwargs)


def run_stage5_tracking(*args, **kwargs):
    from backend.cv.tracking import run_stage5_tracking as _s5
    return _s5(*args, **kwargs)


def validate_observation_record(*args, **kwargs):
    from backend.cv.observations import validate_observation_record as _vor
    return _vor(*args, **kwargs)


def extract_and_save_crop(*args, **kwargs):
    from backend.cv.crops import extract_and_save_crop as _easc
    return _easc(*args, **kwargs)


def validate_crop_file(*args, **kwargs):
    from backend.cv.crops import validate_crop_file as _vcf
    return _vcf(*args, **kwargs)


def run_stage6_observations(*args, **kwargs):
    from backend.cv.observations import run_stage6_observations as _s6
    return _s6(*args, **kwargs)


def run_stage7_integration_audit(*args, **kwargs):
    from backend.cv.integration_audit import run_stage7_integration_audit as _s7
    return _s7(*args, **kwargs)


def generate_person1_handoff_manifest(*args, **kwargs):
    from backend.cv.handoff import generate_person1_handoff_manifest as _gphm
    return _gphm(*args, **kwargs)


__all__ = [
    "discover_video_sources",
    "assign_camera_ids",
    "extract_video_metadata",
    "validate_video_metadata",
    "build_camera_inventory",
    "save_camera_inventory",
    "run_stage1_pipeline",
    "normalize_repo_path",
    "compute_sample_indices",
    "compute_timestamp",
    "run_stage3_sampling",
    "validate_and_clip_bbox",
    "run_stage4_detection",
    "validate_track_record",
    "run_stage5_tracking",
    "validate_observation_record",
    "extract_and_save_crop",
    "validate_crop_file",
    "run_stage6_observations",
    "run_stage7_integration_audit",
    "generate_person1_handoff_manifest",
]


