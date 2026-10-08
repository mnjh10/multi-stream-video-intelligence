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
        output_dir: str | None = None,
    ) -> dict:
        """
        Extract a frame from source video at the exact timestamp.
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

        frame_idx = int(round(timestamp * fps))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Seek to millisecond position
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp * 1000.0))
        success, frame = cap.read()

        if not success or frame is None:
            # Fallback to seeking frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, max(0, total_frames - 1)))
            success, frame = cap.read()

        cap.release()

        if not success or frame is None:
            raise ValueError(
                f"Could not read frame at timestamp {timestamp}s from {source_video}"
            )

        cam_tag = camera_id or "cam"
        evt_tag = event_id or "evt"
        filename = f"{cam_tag}_{evt_tag}_{int(round(timestamp * 1000)):06d}ms.jpg"
        out_path = target_dir / filename

        cv2.imwrite(str(out_path), frame)

        return {
            "status": "ok",
            "frame_path": str(out_path).replace("\\", "/"),
            "timestamp": timestamp,
            "source_video": str(video_path).replace("\\", "/"),
            "camera_id": camera_id,
            "event_id": event_id,
            "frame_index": frame_idx,
            "metadata": {
                "fps": float(fps),
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
                "total_frames": total_frames,
            },
        }
