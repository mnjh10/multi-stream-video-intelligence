import os
from pathlib import Path
import cv2


class EvidenceService:
    """
    Extracts frame or clip evidence from source videos at specified timestamps.
    """

    DEFAULT_OUTPUT_DIR = "data/evidence"

    def __init__(self, default_output_dir: str | None = None):
        self.output_dir = Path(default_output_dir or self.DEFAULT_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frame_evidence(
        self,
        source_video: str,
        timestamp: float,
        camera_id: str | None = None,
        event_id: str | None = None,
        object_id: str | None = None,
        frame_index: int | None = None,
        bbox: list[float] | None = None,
        annotate: bool = True,
        output_dir: str | None = None,
    ) -> dict:
        """
        Extract a frame from source video at the exact timestamp, optionally
        drawing a bounding box around the target object.
        """
        video_path = Path(source_video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {source_video}")

        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {source_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 10.0

        calc_frame_idx = frame_index if frame_index is not None else int(round(timestamp * fps))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Seek to frame: prefer frame_index if provided for keyframe-perfect alignment
        if frame_index is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(calc_frame_idx, max(0, total_frames - 1)))
            success, frame = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp * 1000.0))
            success, frame = cap.read()
            if not success or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(calc_frame_idx, max(0, total_frames - 1)))
                success, frame = cap.read()

        cap.release()

        if not success or frame is None:
            raise ValueError(
                f"Could not read frame at timestamp {timestamp}s from {source_video}"
            )

        h, w = frame.shape[:2]
        cam_tag = camera_id or "cam"
        evt_tag = event_id or "evt"
        ms_tag = int(round(timestamp * 1000))
        
        raw_filename = f"{cam_tag}_{evt_tag}_{ms_tag:06d}ms_raw.jpg"
        annotated_filename = f"{cam_tag}_{evt_tag}_{ms_tag:06d}ms.jpg"

        raw_path = target_dir / raw_filename
        out_path = target_dir / annotated_filename

        # Save raw frame
        cv2.imwrite(str(raw_path), frame)

        # Attempt to resolve bounding box if requested and not provided directly
        resolved_bbox = bbox
        resolved_object_type = "object"
        resolution_msg = ""

        if resolved_bbox is None and annotate:
            if camera_id and object_id:
                try:
                    from backend.services.temporal import get_temporal_followup_engine

                    engine = get_temporal_followup_engine()
                    obs = engine.find_observation(camera_id, object_id, timestamp)
                    if obs:
                        resolved_bbox = obs.get("bbox")
                        if obs.get("object_type"):
                            resolved_object_type = obs.get("object_type")
                        if obs.get("frame_index") is not None and frame_index is None:
                            calc_frame_idx = obs.get("frame_index")
                    else:
                        resolution_msg = (
                            f"Target object '{object_id}' could not be resolved at {timestamp:.2f}s "
                            f"in {camera_id}; returned unannotated raw evidence."
                        )
                except Exception as e:
                    resolution_msg = f"Target resolution failed: {e}"
            else:
                resolution_msg = "Target object or camera ID not specified; returned unannotated raw evidence."

        is_annotated = False
        display_frame = frame.copy()

        # If annotation requested and bounding box is available with 4 coordinates
        if annotate and resolved_bbox and len(resolved_bbox) == 4:
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in resolved_bbox]
                # Clamp coordinates to actual frame dimensions
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                if x2 > x1 and y2 > y1:
                    # Draw contrasting rectangle (BGR: vivid cyan [0, 229, 255])
                    bbox_color = (0, 229, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), bbox_color, 3)

                    # Label format e.g. "car | obj_cam06_000004 | 207.7s"
                    tag_id = object_id or "target"
                    label = f"{resolved_object_type} | {tag_id} | {timestamp:.1f}s"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                    # Position badge safely above bounding box or inside if too close to top
                    if y1 - text_h - 12 < 0:
                        badge_y1 = y1
                        badge_y2 = y1 + text_h + 10
                        text_y = y1 + text_h + 5
                    else:
                        badge_y1 = y1 - text_h - 10
                        badge_y2 = y1
                        text_y = y1 - 4

                    badge_x2 = min(w, x1 + text_w + 12)

                    # Draw dark filled background for contrast and thin cyan outline
                    cv2.rectangle(display_frame, (x1, badge_y1), (badge_x2, badge_y2), (18, 24, 32), -1)
                    cv2.rectangle(display_frame, (x1, badge_y1), (badge_x2, badge_y2), bbox_color, 1)
                    cv2.putText(
                        display_frame,
                        label,
                        (x1 + 6, text_y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
                    is_annotated = True
                    resolution_msg = f"Target object '{tag_id}' localized and annotated with ground bounding box."
            except Exception:
                is_annotated = False

        if is_annotated:
            cv2.imwrite(str(out_path), display_frame)
            final_frame_path = out_path
        else:
            # Fall back to raw frame
            final_frame_path = raw_path
            if not resolution_msg:
                resolution_msg = "Returned unannotated raw evidence frame."

        return {
            "status": "ok",
            "frame_path": str(final_frame_path).replace("\\", "/"),
            "raw_frame_path": str(raw_path).replace("\\", "/"),
            "annotated": is_annotated,
            "message": resolution_msg,
            "timestamp": timestamp,
            "source_video": str(video_path).replace("\\", "/"),
            "camera_id": camera_id,
            "event_id": event_id,
            "object_id": object_id,
            "frame_index": calc_frame_idx,
            "bbox": resolved_bbox if is_annotated else None,
            "metadata": {
                "fps": float(fps),
                "height": int(h),
                "width": int(w),
                "total_frames": total_frames,
            },
        }
