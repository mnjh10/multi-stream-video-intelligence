from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """
    Standard retrieval result returned to downstream systems.
    """

    event_id: str
    camera_id: str
    object_id: str
    timestamp_start: float
    timestamp_end: float
    best_timestamp: float
    score: float
    object_type: str
    source_video: str

    def to_dict(self) -> dict:
        """Convert the result into a JSON-compatible dictionary."""

        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "object_id": self.object_id,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "best_timestamp": self.best_timestamp,
            "score": self.score,
            "object_type": self.object_type,
            "source_video": self.source_video,
        }