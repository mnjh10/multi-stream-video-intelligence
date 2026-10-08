"""Person 1 Stage 5: Multi-Object Tracking & Stable Per-Camera IDs.

Processes offline object detections from Stage 4 using ByteTrack association,
assigning deterministic, camera-local object IDs across consecutive sampled frames.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
from ultralytics.trackers.byte_tracker import BYTETracker, STrack

from backend.cv.discovery import get_repo_root, normalize_repo_path

logger = logging.getLogger("argus.cv.tracking")

DEFAULT_TRACK_HIGH_THRESH: float = 0.25
DEFAULT_TRACK_LOW_THRESH: float = 0.1
DEFAULT_NEW_TRACK_THRESH: float = 0.25
DEFAULT_TRACK_BUFFER: int = 30
DEFAULT_MATCH_THRESH: float = 0.8
DEFAULT_FUSE_SCORE: bool = True

OBJECT_ID_PATTERN = re.compile(r"^obj_cam\d{2}_\d{6}$")


@dataclass
class TrackingConfig:
    """Configuration parameters for Stage 5 multi-object tracking."""

    input_detections: str = "data/detections/detections.jsonl"
    frame_index_path: str = "data/frames/frame_index.jsonl"
    output_dir: str = "data/tracks"
    tracks_file: str = "data/tracks/tracks.jsonl"
    manifest_file: str = "data/tracks/track_manifest.json"
    track_high_thresh: float = DEFAULT_TRACK_HIGH_THRESH
    track_low_thresh: float = DEFAULT_TRACK_LOW_THRESH
    new_track_thresh: float = DEFAULT_NEW_TRACK_THRESH
    track_buffer: int = DEFAULT_TRACK_BUFFER
    match_thresh: float = DEFAULT_MATCH_THRESH
    fuse_score: bool = DEFAULT_FUSE_SCORE

    def __post_init__(self) -> None:
        """Validate tracking configuration parameters."""
        if not (0.0 < self.track_high_thresh <= 1.0):
            raise ValueError(f"track_high_thresh must be in (0.0, 1.0], got {self.track_high_thresh}")
        if not (0.0 <= self.track_low_thresh <= self.track_high_thresh):
            raise ValueError(f"track_low_thresh must be in [0.0, track_high_thresh], got {self.track_low_thresh}")
        if not (0.0 < self.new_track_thresh <= 1.0):
            raise ValueError(f"new_track_thresh must be in (0.0, 1.0], got {self.new_track_thresh}")
        if self.track_buffer < 1:
            raise ValueError(f"track_buffer must be >= 1, got {self.track_buffer}")
        if not (0.0 < self.match_thresh <= 1.0):
            raise ValueError(f"match_thresh must be in (0.0, 1.0], got {self.match_thresh}")


class OfflineDetections:
    """Lightweight adapter exposing Stage 4 offline detections to BYTETracker.

    Exposes .xyxy, .conf, .cls, .xywh, supports len() and boolean mask slicing.
    """

    def __init__(
        self,
        xyxy: list[list[float]] | np.ndarray,
        conf: list[float] | np.ndarray,
        cls: list[int] | np.ndarray,
    ) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4) if len(xyxy) > 0 else np.zeros((0, 4), dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1) if len(conf) > 0 else np.zeros((0,), dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1) if len(cls) > 0 else np.zeros((0,), dtype=np.float32)

        if len(self.xyxy) > 0:
            w = self.xyxy[:, 2] - self.xyxy[:, 0]
            h = self.xyxy[:, 3] - self.xyxy[:, 1]
            cx = self.xyxy[:, 0] + w / 2.0
            cy = self.xyxy[:, 1] + h / 2.0
            self.xywh = np.stack([cx, cy, w, h], axis=-1)
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx: Any) -> OfflineDetections:
        return OfflineDetections(self.xyxy[idx], self.conf[idx], self.cls[idx])


def format_object_id(camera_id: str, numeric_id: int) -> str:
    """Format camera-local deterministic object ID.

    Example: 'cam_01', 1 -> 'obj_cam01_000001'
    """
    clean_cam = camera_id.replace("_", "")
    return f"obj_{clean_cam}_{numeric_id:06d}"


def validate_track_record(record: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate all fields and constraints of an individual track record."""
    required_keys = [
        "camera_id",
        "frame_index",
        "timestamp",
        "object_id",
        "object_type",
        "class_id",
        "confidence",
        "bbox",
        "frame_path",
        "source_video",
    ]
    for key in required_keys:
        if key not in record:
            return False, f"Missing required key: '{key}'"

    obj_id = record["object_id"]
    if not isinstance(obj_id, str) or not OBJECT_ID_PATTERN.match(obj_id):
        return False, f"Invalid object_id format: '{obj_id}'"

    clean_cam = record["camera_id"].replace("_", "")
    if clean_cam not in obj_id:
        return False, f"object_id '{obj_id}' does not match camera_id '{record['camera_id']}'"

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

    return True, None


def track_camera_detections(
    camera_id: str,
    frames_dict: dict[int, list[dict[str, Any]]],
    config: TrackingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute ByteTrack association over a single camera's chronological detections.

    Args:
        camera_id: Canonical camera identifier (e.g. 'cam_01').
        frames_dict: Map of frame_index -> list of Stage 4 detection records.
        config: Tracking configuration parameters.

    Returns:
        Tuple of (tracked_records_list, camera_tracking_summary_stats).
    """
    STrack.reset_id()
    tracker_args = SimpleNamespace(
        track_high_thresh=config.track_high_thresh,
        track_low_thresh=config.track_low_thresh,
        new_track_thresh=config.new_track_thresh,
        track_buffer=config.track_buffer,
        match_thresh=config.match_thresh,
        fuse_score=config.fuse_score,
    )
    tracker = BYTETracker(tracker_args)

    track_first_det: dict[int, tuple[int, dict[str, Any]]] = {}
    confirmed_tracks: set[int] = set()
    raw_track_records: list[tuple[int, int, dict[str, Any]]] = []  # (frame_index, raw_tid, detection)

    sorted_frame_indices = sorted(frames_dict.keys())

    for fidx in sorted_frame_indices:
        dets = frames_dict[fidx]
        if not dets:
            d_obj = OfflineDetections([], [], [])
            tracker.update(d_obj)
            continue

        d_obj = OfflineDetections(
            xyxy=[d["bbox"] for d in dets],
            conf=[d["confidence"] for d in dets],
            cls=[d["class_id"] for d in dets],
        )
        tracker.update(d_obj)

        for strack in tracker.tracked_stracks:
            tid = strack.track_id
            det_idx = int(strack.idx)
            matched_det = dets[det_idx]

            if tid not in track_first_det:
                # Record the initial detection that seeded this tracklet
                track_first_det[tid] = (fidx, matched_det)

            if strack.is_activated:
                if tid not in confirmed_tracks:
                    confirmed_tracks.add(tid)
                    # If confirmed on a frame later than initial appearance, record initial frame
                    init_fidx, init_det = track_first_det[tid]
                    if init_fidx != fidx:
                        raw_track_records.append((init_fidx, tid, init_det))

                raw_track_records.append((fidx, tid, matched_det))

    # Deterministic mapping: sort confirmed tracks by (first_frame_index, first_detection_id)
    sorted_confirmed = sorted(
        confirmed_tracks,
        key=lambda tid: (track_first_det[tid][0], track_first_det[tid][1].get("detection_id", "")),
    )
    raw_to_obj_id = {
        tid: format_object_id(camera_id, num)
        for num, tid in enumerate(sorted_confirmed, start=1)
    }

    # Deduplicate and format output records
    # (frame_index, object_id) must be strictly unique per camera frame
    seen_frame_obj: set[tuple[int, str]] = set()
    output_records: list[dict[str, Any]] = []
    tracked_det_ids: set[str] = set()

    for fidx, raw_tid, det in raw_track_records:
        obj_id = raw_to_obj_id[raw_tid]
        key = (fidx, obj_id)
        if key in seen_frame_obj:
            continue
        seen_frame_obj.add(key)

        det_id = det.get("detection_id", "")
        if det_id:
            tracked_det_ids.add(det_id)

        rec = {
            "camera_id": camera_id,
            "frame_index": fidx,
            "timestamp": det["timestamp"],
            "object_id": obj_id,
            "object_type": det["class_name"],
            "class_id": det["class_id"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "frame_path": det["frame_path"],
            "source_video": det["source_video"],
            "detection_id": det_id,
        }
        output_records.append(rec)

    # Sort deterministically: frame_index ascending, then object_id ascending
    output_records.sort(key=lambda r: (r["frame_index"], r["object_id"]))

    total_input_dets = sum(len(dets) for dets in frames_dict.values())
    tracked_count = len(output_records)
    discarded_count = total_input_dets - len(tracked_det_ids)

    summary = {
        "camera_id": camera_id,
        "input_detections": total_input_dets,
        "tracked_detections": tracked_count,
        "discarded_detections": discarded_count,
        "unique_objects": len(sorted_confirmed),
        "processed_frames": len(sorted_frame_indices),
    }

    return output_records, summary


def run_stage5_tracking(
    input_detections: Path | str = "data/detections/detections.jsonl",
    frame_index_path: Path | str = "data/frames/frame_index.jsonl",
    output_dir: Path | str = "data/tracks",
    track_buffer: int = DEFAULT_TRACK_BUFFER,
    match_thresh: float = DEFAULT_MATCH_THRESH,
    repo_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Execute Stage 5 tracking pipeline across all 11 CityFlowV2 cameras.

    Args:
        input_detections: Path to Stage 4 detections file.
        frame_index_path: Path to Stage 3 authoritative frame index.
        output_dir: Directory for serialized tracks and manifest.
        track_buffer: Maximum frames to retain lost tracks.
        match_thresh: Association IoU matching threshold.
        repo_root: Optional root directory for path resolution.

    Returns:
        Summary manifest dictionary with tracking metrics.
    """
    root = Path(repo_root).resolve() if repo_root else get_repo_root()
    config = TrackingConfig(
        input_detections=str(input_detections),
        frame_index_path=str(frame_index_path),
        output_dir=str(output_dir),
        tracks_file=f"{output_dir}/tracks.jsonl".replace("\\", "/"),
        manifest_file=f"{output_dir}/track_manifest.json".replace("\\", "/"),
        track_buffer=track_buffer,
        match_thresh=match_thresh,
    )

    det_path = root / config.input_detections
    if not det_path.exists():
        raise FileNotFoundError(f"Input detections file not found: {det_path}")

    fi_path = root / config.frame_index_path
    if not fi_path.exists():
        raise FileNotFoundError(f"Authoritative frame index not found: {fi_path}")

    # 1. Load authoritative frame index to guarantee every camera and frame is registered
    authoritative_frames: dict[str, set[int]] = {}
    with open(fi_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["camera_id"]
            if cid not in authoritative_frames:
                authoritative_frames[cid] = set()
            authoritative_frames[cid].add(rec["frame_index"])

    all_cameras = sorted(authoritative_frames.keys())

    # 2. Load detections grouped by camera_id -> frame_index -> list[det]
    camera_frames: dict[str, dict[int, list[dict[str, Any]]]] = {
        cid: {fidx: [] for fidx in sorted(authoritative_frames[cid])}
        for cid in all_cameras
    }

    total_input_detections = 0
    with open(det_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["camera_id"]
            fidx = rec["frame_index"]
            if cid in camera_frames and fidx in camera_frames[cid]:
                camera_frames[cid][fidx].append(rec)
            total_input_detections += 1

    out_dir = root / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = root / config.tracks_file
    manifest_path = root / config.manifest_file

    print("=" * 65)
    print("STAGE 5 — OBJECT TRACKING & STABLE PER-CAMERA IDS (BYTETRACK)")
    print(f"Input Detections: {normalize_repo_path(det_path, root)} ({total_input_detections} records)")
    print(f"Total Cameras: {len(all_cameras)} | Total Frames: {sum(len(v) for v in authoritative_frames.values())}")
    print(f"Track Buffer: {config.track_buffer} | Match Threshold: {config.match_thresh}")
    print("=" * 65)

    all_tracked_records: list[dict[str, Any]] = []
    objects_per_camera: dict[str, int] = {}
    detections_per_camera: dict[str, int] = {}
    tracked_per_camera: dict[str, int] = {}
    discarded_per_camera: dict[str, int] = {}
    frames_per_camera: dict[str, int] = {}
    class_distribution: dict[str, int] = {}

    t_start = time.time()

    with open(tracks_path, "w", encoding="utf-8") as out_f:
        for cid in all_cameras:
            c_frames = camera_frames[cid]
            c_records, c_summary = track_camera_detections(cid, c_frames, config)

            # Defensive validation of all output records
            for rec in c_records:
                is_valid, err = validate_track_record(rec)
                if not is_valid:
                    raise ValueError(f"Track record validation failed for {rec['object_id']}: {err}")
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                all_tracked_records.append(rec)

                obj_type = rec["object_type"]
                class_distribution[obj_type] = class_distribution.get(obj_type, 0) + 1

            objects_per_camera[cid] = c_summary["unique_objects"]
            detections_per_camera[cid] = c_summary["input_detections"]
            tracked_per_camera[cid] = c_summary["tracked_detections"]
            discarded_per_camera[cid] = c_summary["discarded_detections"]
            frames_per_camera[cid] = len(c_frames)

            pct = (c_summary["tracked_detections"] / c_summary["input_detections"] * 100) if c_summary["input_detections"] > 0 else 0
            print(
                f"  {cid}: {c_summary['unique_objects']:4d} tracks | "
                f"{c_summary['tracked_detections']:5d}/{c_summary['input_detections']:5d} tracked ({pct:5.1f}%) | "
                f"{c_summary['discarded_detections']:4d} unconfirmed discarded"
            )

    t_end = time.time()
    tracking_time = round(t_end - t_start, 3)
    total_frames = sum(frames_per_camera.values())
    fps = round(total_frames / tracking_time, 1) if tracking_time > 0 else 0
    avg_ms = round((tracking_time / total_frames) * 1000, 2) if total_frames > 0 else 0

    total_tracked_detections = len(all_tracked_records)
    total_discarded = total_input_detections - total_tracked_detections
    total_unique_objects = sum(objects_per_camera.values())

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "stage": "Stage 5 — Multi-Object Tracking & Stable Per-Camera IDs",
        "tracker_name": "ByteTrack",
        "tracker_version": "8.4",
        "tracker_backend": "ultralytics.trackers.byte_tracker.BYTETracker",
        "configuration": {
            "track_high_thresh": config.track_high_thresh,
            "track_low_thresh": config.track_low_thresh,
            "new_track_thresh": config.new_track_thresh,
            "track_buffer": config.track_buffer,
            "match_thresh": config.match_thresh,
            "fuse_score": config.fuse_score,
        },
        "input_detection_file": normalize_repo_path(det_path, root),
        "input_detection_count": total_input_detections,
        "camera_count": len(all_cameras),
        "frame_count": total_frames,
        "tracked_detection_count": total_tracked_detections,
        "discarded_detection_count": total_discarded,
        "unique_object_count": total_unique_objects,
        "objects_per_camera": objects_per_camera,
        "detections_per_camera": detections_per_camera,
        "tracked_per_camera": tracked_per_camera,
        "discarded_per_camera": discarded_per_camera,
        "frames_per_camera": frames_per_camera,
        "class_distribution": dict(sorted(class_distribution.items(), key=lambda x: -x[1])),
        "performance": {
            "tracking_time_seconds": tracking_time,
            "frames_per_second": fps,
            "avg_ms_per_frame": avg_ms,
        },
        "tracks_file": normalize_repo_path(tracks_path, root),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print(f"TRACKING COMPLETE: {total_unique_objects} unique objects tracked ({total_tracked_detections} records).")
    print(f"Discarded unconfirmed: {total_discarded} detections.")
    print(f"Runtime: {tracking_time}s ({fps} fps, {avg_ms} ms/frame)")
    print(f"Tracks JSONL: {normalize_repo_path(tracks_path, root)}")
    print(f"Manifest JSON: {normalize_repo_path(manifest_path, root)}")
    print("=" * 65)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person 1 Stage 5 ByteTrack Tracking")
    parser.add_argument("--detections", default="data/detections/detections.jsonl", help="Input detections file")
    parser.add_argument("--frame-index", default="data/frames/frame_index.jsonl", help="Authoritative frame index")
    parser.add_argument("--output-dir", default="data/tracks", help="Output directory for tracks")
    parser.add_argument("--track-buffer", type=int, default=DEFAULT_TRACK_BUFFER, help="Track buffer length")
    parser.add_argument("--match-thresh", type=float, default=DEFAULT_MATCH_THRESH, help="Matching IoU threshold")
    args = parser.parse_args()

    run_stage5_tracking(
        input_detections=args.detections,
        frame_index_path=args.frame_index,
        output_dir=args.output_dir,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
    )
