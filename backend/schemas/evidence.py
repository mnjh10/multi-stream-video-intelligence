from pydantic import BaseModel


class EvidenceRequest(BaseModel):
    source_video: str
    timestamp: float
    event_id: str


class EvidenceResponse(BaseModel):
    frame_path: str
    clip_path: str
    start_time: float
    end_time: float