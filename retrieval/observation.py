from dataclasses import dataclass


@dataclass
class Observation:
    """
    Represents one object observation produced by Person 1's
    computer-vision pipeline.
    """

    observation_id: str
    camera_id: str
    timestamp: float
    frame_index: int
    object_id: str
    object_type: str
    confidence: float
    bbox: list[float]
    frame_path: str
    crop_path: str
    source_video: str

    def to_dict(self) -> dict:
        """Convert the observation into a dictionary."""

        return {
            "observation_id": self.observation_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "object_id": self.object_id,
            "object_type": self.object_type,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "frame_path": self.frame_path,
            "crop_path": self.crop_path,
            "source_video": self.source_video,
        }