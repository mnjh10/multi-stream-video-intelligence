"""YOLO11 Object Detection and Bounding-Box Validation for Person 1 CV Pipeline.

Processes sampled video frames from Stage 3, applies Ultralytics YOLO11 inference,
validates and clips bounding boxes to native image boundaries, and generates
deterministic detection records.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import torch
from ultralytics import YOLO

from backend.cv.discovery import get_repo_root, normalize_repo_path

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME: str = "yolo11n.pt"
DEFAULT_CONF_THRESHOLD: float = 0.25
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_MIN_BOX_SIZE: float = 1.0


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration parameters for YOLO11 object detection."""

    model_name: str = DEFAULT_MODEL_NAME
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    batch_size: int = DEFAULT_BATCH_SIZE
    min_box_size: float = DEFAULT_MIN_BOX_SIZE
    device: Optional[str] = None
    output_dir: str = "data/detections"
    detections_file: str = "data/detections/detections.jsonl"
    manifest_file: str = "data/detections/detection_manifest.json"
    clip_boxes: bool = True

    def __post_init__(self):
        if not (0.0 < self.conf_threshold <= 1.0):
            raise ValueError(f"conf_threshold must be in (0.0, 1.0], got {self.conf_threshold}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.min_box_size < 0.0:
            raise ValueError(f"min_box_size must be >= 0.0, got {self.min_box_size}")


def validate_and_clip_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    clip: bool = True,
    min_size: float = DEFAULT_MIN_BOX_SIZE,
) -> tuple[bool, Optional[list[float]], Optional[str]]:
    """Validate and optionally clip a bounding box to native image boundaries.

    Policy:
        Clip coordinates to [0, image_width] and [0, image_height], then validate
        that 0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height with non-zero area.

    Args:
        bbox: Coordinate quadruple [x1, y1, x2, y2].
        image_width: Width of native source image.
        image_height: Height of native source image.
        clip: If True, clip coordinates to image boundaries.
        min_size: Minimum permissible width and height in pixels.

    Returns:
        (is_valid, clipped_bbox, error_reason)
    """
    if len(bbox) != 4:
        return False, None, f"Bounding box must contain exactly 4 coordinates, got {len(bbox)}"

    x1, y1, x2, y2 = bbox

    # Check finite numeric values
    for val, name in zip([x1, y1, x2, y2], ["x1", "y1", "x2", "y2"]):
        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            return False, None, f"Coordinate {name} is not a finite number: {val}"

    w_max = float(image_width)
    h_max = float(image_height)

    if clip:
        x1_c = max(0.0, min(float(x1), w_max))
        y1_c = max(0.0, min(float(y1), h_max))
        x2_c = max(0.0, min(float(x2), w_max))
        y2_c = max(0.0, min(float(y2), h_max))
    else:
        x1_c, y1_c, x2_c, y2_c = float(x1), float(y1), float(x2), float(y2)

    # Validate coordinate ordering and minimum dimensions
    box_w = x2_c - x1_c
    box_h = y2_c - y1_c

    if box_w < min_size:
        return False, None, f"Box width ({box_w:.2f}) is below minimum size ({min_size})"
    if box_h < min_size:
        return False, None, f"Box height ({box_h:.2f}) is below minimum size ({min_size})"
    if x1_c < 0.0 or x2_c > w_max or y1_c < 0.0 or y2_c > h_max:
        return False, None, f"Box [{x1_c}, {y1_c}, {x2_c}, {y2_c}] exceeds boundaries [0, 0, {w_max}, {h_max}]"

    rounded_box = [round(x1_c, 2), round(y1_c, 2), round(x2_c, 2), round(y2_c, 2)]
    return True, rounded_box, None


def validate_detection_record(record: dict[str, Any], image_width: int, image_height: int) -> tuple[bool, Optional[str]]:
    """Validate all fields of an individual detection record."""
    required_keys = [
        "detection_id",
        "camera_id",
        "frame_index",
        "timestamp",
        "class_id",
        "class_name",
        "confidence",
        "bbox",
        "frame_path",
        "source_video",
    ]
    for key in required_keys:
        if key not in record:
            return False, f"Missing required key: '{key}'"

    if not isinstance(record["detection_id"], str) or not record["detection_id"].startswith("det_"):
        return False, f"Invalid detection_id format: {record['detection_id']}"

    if not isinstance(record["frame_index"], int) or record["frame_index"] < 0:
        return False, f"Invalid frame_index: {record['frame_index']}"

    ts = record["timestamp"]
    if not isinstance(ts, (int, float)) or math.isnan(ts) or math.isinf(ts) or ts < 0:
        return False, f"Invalid timestamp: {ts}"

    conf = record["confidence"]
    if not isinstance(conf, (int, float)) or math.isnan(conf) or math.isinf(conf) or not (0.0 <= conf <= 1.0):
        return False, f"Invalid confidence: {conf}"

    if not isinstance(record["class_id"], int) or record["class_id"] < 0:
        return False, f"Invalid class_id: {record['class_id']}"

    if not isinstance(record["class_name"], str) or not record["class_name"].strip():
        return False, f"Invalid class_name: {record['class_name']}"

    valid_box, _, err = validate_and_clip_bbox(
        bbox=record["bbox"],
        image_width=image_width,
        image_height=image_height,
        clip=False,  # Already clipped
        min_size=0.1,
    )
    if not valid_box:
        return False, f"Invalid bbox: {err}"

    return True, None


def load_authoritative_frame_index(
    frame_index_path: Path | str,
    repo_root: Optional[Path | str] = None,
) -> list[dict[str, Any]]:
    """Load authoritative frame records from Stage 3 frame_index.jsonl.

    Args:
        frame_index_path: Path to frame_index.jsonl.
        repo_root: Optional repository root for path resolution.

    Returns:
        List of frame index dictionaries sorted deterministically by (camera_id, frame_index).
    """
    root = Path(repo_root).resolve() if repo_root else get_repo_root()
    index_file = root / frame_index_path if not Path(frame_index_path).is_absolute() else Path(frame_index_path)

    if not index_file.exists():
        raise FileNotFoundError(f"Authoritative frame index not found: {index_file}")

    records = []
    with open(index_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)

    # Enforce deterministic order
    records = sorted(records, key=lambda r: (r["camera_id"].lower(), r["camera_id"], r["frame_index"]))
    return records


def run_stage4_detection(
    frame_index_path: Path | str = "data/frames/frame_index.jsonl",
    output_dir: Path | str = "data/detections",
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    repo_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Execute Stage 4 YOLO11 object detection pipeline across all sampled frames.

    Args:
        frame_index_path: Path to authoritative frame index from Stage 3.
        output_dir: Target directory for detection records and manifest.
        conf_threshold: Confidence threshold for candidate detections.
        batch_size: Number of images per inference batch.
        model_name: Ultralytics model weights identifier or path.
        device: Target inference device ('cuda:0', 'cpu', or None for auto).
        repo_root: Repository root path for path resolution.

    Returns:
        Summary dictionary with execution metrics and detection statistics.
    """
    root = Path(repo_root).resolve() if repo_root else get_repo_root()
    config = DetectionConfig(
        model_name=model_name,
        conf_threshold=conf_threshold,
        batch_size=batch_size,
        device=device,
        output_dir=str(Path(output_dir).as_posix()),
        detections_file=f"{output_dir}/detections.jsonl".replace("\\", "/"),
        manifest_file=f"{output_dir}/detection_manifest.json".replace("\\", "/"),
    )

    # 1. Determine device
    if config.device:
        target_device = config.device
    else:
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and "cuda" in target_device else None

    # 2. Load model
    t_model_start = time.time()
    model = YOLO(config.model_name)
    model_load_time = round(time.time() - t_model_start, 3)

    # 3. Load authoritative frames
    frame_records = load_authoritative_frame_index(frame_index_path, repo_root=root)
    total_frames = len(frame_records)
    if total_frames == 0:
        raise RuntimeError("No frames found in authoritative frame index")

    out_dir = root / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detections_path = root / config.detections_file
    manifest_path = root / config.manifest_file

    print("=" * 65)
    print("STAGE 4 — YOLO11 OBJECT DETECTION & BOUNDING-BOX VALIDATION")
    print(f"Model: {config.model_name} (Loaded in {model_load_time}s)")
    print(f"Device: {target_device} ({gpu_name or 'CPU'})")
    print(f"Confidence Threshold: {config.conf_threshold} | Batch Size: {config.batch_size}")
    print(f"Total Authoritative Frames: {total_frames}")
    print("=" * 65)

    # 4. Image dimension cache per camera (to avoid re-reading image headers unnecessarily)
    camera_dims: dict[str, tuple[int, int]] = {}

    all_detections: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    camera_counts: dict[str, int] = {}
    confidences: list[float] = []

    global_det_id = 0
    t_infer_start = time.time()
    processed_frames = 0
    failed_frames = 0

    # Process in bounded batches
    num_batches = (total_frames + config.batch_size - 1) // config.batch_size

    with open(detections_path, "w", encoding="utf-8") as out_f:
        for b_idx in range(num_batches):
            b_start = b_idx * config.batch_size
            b_end = min(b_start + config.batch_size, total_frames)
            batch_records = frame_records[b_start:b_end]

            batch_paths = []
            valid_batch_records = []
            for r in batch_records:
                abs_frame_path = root / r["frame_path"]
                if not abs_frame_path.exists():
                    logger.error(f"Frame missing: {abs_frame_path}")
                    failed_frames += 1
                    continue
                batch_paths.append(str(abs_frame_path))
                valid_batch_records.append(r)

            if not batch_paths:
                continue

            # Run inference on batch
            results = model(
                batch_paths,
                batch=len(batch_paths),
                conf=config.conf_threshold,
                device=target_device,
                verbose=False,
            )

            # Process results deterministically
            for frame_meta, res in zip(valid_batch_records, results):
                processed_frames += 1
                cam_id = frame_meta["camera_id"]
                orig_shape = res.orig_shape  # (height, width)
                img_h, img_w = int(orig_shape[0]), int(orig_shape[1])
                camera_dims[cam_id] = (img_w, img_h)

                boxes = res.boxes
                frame_candidates: list[dict[str, Any]] = []

                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        box = boxes[i]
                        cls_id = int(box.cls[0].item())
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()

                        valid_box, clipped_xyxy, err_reason = validate_and_clip_bbox(
                            bbox=xyxy,
                            image_width=img_w,
                            image_height=img_h,
                            clip=config.clip_boxes,
                            min_size=config.min_box_size,
                        )
                        if not valid_box:
                            logger.debug(f"Rejected box in {frame_meta['frame_path']}: {err_reason}")
                            continue

                        cand = {
                            "camera_id": cam_id,
                            "frame_index": frame_meta["frame_index"],
                            "timestamp": frame_meta["timestamp"],
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(conf, 4),
                            "bbox": clipped_xyxy,
                            "frame_path": frame_meta["frame_path"],
                            "source_video": frame_meta["source_video"],
                        }
                        frame_candidates.append(cand)

                # Sort detections within each frame deterministically
                frame_candidates.sort(
                    key=lambda d: (
                        d["class_id"],
                        d["bbox"][0],
                        d["bbox"][1],
                        d["bbox"][2],
                        d["bbox"][3],
                        -d["confidence"],
                    )
                )

                # Assign sequential global detection_id and serialize
                for cand in frame_candidates:
                    global_det_id += 1
                    det_id = f"det_{global_det_id:06d}"
                    cand["detection_id"] = det_id

                    # Defensive schema validation
                    is_valid, val_err = validate_detection_record(cand, image_width=img_w, image_height=img_h)
                    if not is_valid:
                        raise ValueError(f"Detection validation failed for {det_id}: {val_err}")

                    # Write line-delimited JSON
                    out_f.write(json.dumps(cand, ensure_ascii=False) + "\n")

                    # Aggregate statistics
                    c_name = cand["class_name"]
                    class_counts[c_name] = class_counts.get(c_name, 0) + 1
                    camera_counts[cam_id] = camera_counts.get(cam_id, 0) + 1
                    confidences.append(cand["confidence"])

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == num_batches:
                elapsed_so_far = time.time() - t_infer_start
                cur_fps = processed_frames / elapsed_so_far if elapsed_so_far > 0 else 0
                print(
                    f"  Batch {b_idx + 1}/{num_batches}: {processed_frames}/{total_frames} frames "
                    f"({cur_fps:.1f} fps) | Detections: {global_det_id}"
                )

    t_infer_end = time.time()
    inference_time = round(t_infer_end - t_infer_start, 3)
    avg_fps = round(processed_frames / inference_time, 1) if inference_time > 0 else 0
    avg_time_per_frame_ms = round((inference_time / processed_frames) * 1000, 2) if processed_frames > 0 else 0

    conf_mean = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    conf_min = round(min(confidences), 4) if confidences else 0.0
    conf_max = round(max(confidences), 4) if confidences else 0.0

    # Deterministic summary manifest
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "model_name": config.model_name,
        "ultralytics_version": getattr(model, "__version__", "8.4"),
        "torch_version": torch.__version__,
        "device": target_device,
        "gpu_name": gpu_name,
        "conf_threshold": config.conf_threshold,
        "batch_size": config.batch_size,
        "total_input_frames": total_frames,
        "processed_frames": processed_frames,
        "failed_frames": failed_frames,
        "total_detections": global_det_id,
        "confidence_statistics": {
            "min": conf_min,
            "max": conf_max,
            "mean": conf_mean,
        },
        "frames_per_camera": {cam: sum(1 for r in frame_records if r["camera_id"] == cam) for cam in sorted(set(r["camera_id"] for r in frame_records))},
        "detections_per_camera": {cam: camera_counts.get(cam, 0) for cam in sorted(set(r["camera_id"] for r in frame_records))},
        "class_distribution": dict(sorted(class_counts.items(), key=lambda x: -x[1])),
        "performance": {
            "model_load_time_seconds": model_load_time,
            "inference_time_seconds": inference_time,
            "frames_per_second": avg_fps,
            "avg_time_per_frame_ms": avg_time_per_frame_ms,
        },
        "detections_file": normalize_repo_path(detections_path, root),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print(f"DETECTION COMPLETE: {global_det_id} objects detected across {processed_frames} frames.")
    print(f"Inference time: {inference_time}s ({avg_fps} fps, {avg_time_per_frame_ms} ms/frame)")
    print(f"Mean confidence: {conf_mean} (min={conf_min}, max={conf_max})")
    print(f"Detections JSONL: {normalize_repo_path(detections_path, root)}")
    print(f"Manifest JSON: {normalize_repo_path(manifest_path, root)}")
    print("=" * 65)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person 1 YOLO11 Object Detection")
    parser.add_argument("--frame-index", default="data/frames/frame_index.jsonl", help="Path to Stage 3 frame index")
    parser.add_argument("--output-dir", default="data/detections", help="Output directory for detections")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold (default 0.25)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="YOLO11 model identifier or path")
    parser.add_argument("--device", default=None, help="Device to use ('cuda:0' or 'cpu')")
    args = parser.parse_args()

    run_stage4_detection(
        frame_index_path=args.frame_index,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        batch_size=args.batch_size,
        model_name=args.model,
        device=args.device,
    )
