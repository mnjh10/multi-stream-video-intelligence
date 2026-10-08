from pathlib import Path
import subprocess

import cv2


class EvidenceService:
    def __init__(
        self,
        output_dir: str = "evidence",
        clip_window: float = 5.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clip_window = clip_window

    def generate_evidence(
        self,
        source_video: str,
        timestamp: float,
        event_id: str,
    ) -> dict:
        video_path = Path(source_video)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Source video not found: {source_video}"
            )

        frame_path = self.output_dir / f"{event_id}_frame.jpg"
        clip_path = self.output_dir / f"{event_id}_clip.mp4"

        self._extract_frame(
            source_video,
            timestamp,
            frame_path,
        )

        start_time = max(0.0, timestamp - self.clip_window)
        end_time = timestamp + self.clip_window

        self._extract_clip(
            source_video,
            start_time,
            end_time,
            clip_path,
        )

        return {
            "frame_path": str(frame_path),
            "clip_path": str(clip_path),
            "start_time": start_time,
            "end_time": end_time,
        }

    def _extract_frame(
        self,
        source_video: str,
        timestamp: float,
        output_path: Path,
    ) -> None:
        cap = cv2.VideoCapture(source_video)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {source_video}"
            )

        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

        success, frame = cap.read()
        cap.release()

        if not success:
            raise RuntimeError(
                f"Could not extract frame at {timestamp} seconds"
            )

        cv2.imwrite(str(output_path), frame)

    def _extract_clip(
        self,
        source_video: str,
        start_time: float,
        end_time: float,
        output_path: Path,
    ) -> None:
        duration = end_time - start_time

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            source_video,
            "-t",
            str(duration),
            "-c",
            "copy",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed: {result.stderr}"
            )