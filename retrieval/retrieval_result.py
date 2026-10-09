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
    observation_id: str | None = None
    frame_index: int | None = None
    bbox: list[float] | None = None
    crop_path: str | None = None
    verification_status: str | None = None
    attribute_details: dict | None = None

    def to_dict(self) -> dict:
        """Convert the result into a JSON-compatible dictionary."""

        d = {
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
        if self.observation_id is not None:
            d["observation_id"] = self.observation_id
        if self.frame_index is not None:
            d["frame_index"] = self.frame_index
        if self.bbox is not None:
            d["bbox"] = self.bbox
        if self.crop_path is not None:
            d["crop_path"] = self.crop_path
        if self.verification_status is not None:
            d["verification_status"] = self.verification_status
        if self.attribute_details is not None:
            d["attribute_details"] = self.attribute_details
        return d