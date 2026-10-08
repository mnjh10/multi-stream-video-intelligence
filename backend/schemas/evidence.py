from pydantic import BaseModel, Field


class EvidenceRequest(BaseModel):
    source_video: str = Field(description="Path to source video file")
    timestamp: float = Field(ge=0.0, description="Timestamp in seconds")
    camera_id: str | None = Field(default=None, description="Camera identifier")
    event_id: str | None = Field(default=None, description="Associated event ID")
    output_dir: str | None = Field(default=None, description="Optional directory to save extracted evidence")


class EvidenceResponse(BaseModel):
    status: str
    frame_path: str
    timestamp: float
    source_video: str
    camera_id: str | None = None
    event_id: str | None = None
    frame_index: int | None = None
    metadata: dict | None = None
