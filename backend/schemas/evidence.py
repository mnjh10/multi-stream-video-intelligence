from pydantic import BaseModel, Field


class EvidenceRequest(BaseModel):
    source_video: str = Field(description="Path to source video file")
    timestamp: float = Field(ge=0.0, description="Timestamp in seconds")
    camera_id: str | None = Field(default=None, description="Camera identifier")
    event_id: str | None = Field(default=None, description="Associated event ID")
    object_id: str | None = Field(default=None, description="Associated object ID")
    frame_index: int | None = Field(default=None, description="Target frame index")
    bbox: list[float] | None = Field(default=None, description="Target bounding box [x1, y1, x2, y2]")
    annotate: bool = Field(default=True, description="Whether to draw bounding box annotation")
    output_dir: str | None = Field(default=None, description="Optional directory to save extracted evidence")


class EvidenceResponse(BaseModel):
    status: str
    frame_path: str
    timestamp: float
    source_video: str
    camera_id: str | None = None
    event_id: str | None = None
    object_id: str | None = None
    frame_index: int | None = None
    bbox: list[float] | None = None
    annotated: bool = False
    raw_frame_path: str | None = None
    message: str | None = None
    metadata: dict | None = None
