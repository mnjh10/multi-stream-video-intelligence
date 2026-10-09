from datetime import datetime

from pydantic import BaseModel, Field


class QueryContext(BaseModel):
    camera_id: str | None = None
    object_id: str | None = None
    timestamp: float | None = None
    event_id: str | None = None
    frame_index: int | None = None
    bbox: list[float] | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)
    filters: dict | None = None
    context: QueryContext | None = None


class Evidence(BaseModel):
    frame_path: str | None = None
    clip_path: str | None = None
    source_video: str | None = None


class QueryResult(BaseModel):
    event_id: str
    camera_id: str
    timestamp: datetime | float | str
    best_timestamp: float | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    score: float | None = None
    object_id: str | None = None
    object_type: str | None = None
    evidence: Evidence | None = None
    bbox: list[float] | None = None
    frame_index: int | None = None
    crop_path: str | None = None
    observation_id: str | None = None
    verification_status: str | None = None
    attribute_details: dict | None = None
    is_followup: bool = False
    followup_relation: str | None = None


class QueryResponse(BaseModel):
    query: str
    status: str
    message: str | None = None
    results: list[QueryResult] = []