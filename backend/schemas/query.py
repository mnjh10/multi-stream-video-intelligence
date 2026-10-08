from datetime import datetime

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class Evidence(BaseModel):
    frame_path: str | None = None
    clip_path: str | None = None


class QueryResult(BaseModel):
    event_id: str
    camera_id: str
    timestamp: datetime
    score: float | None = None
    object_type: str | None = None
    evidence: Evidence | None = None


class QueryResponse(BaseModel):
    query: str
    status: str
    results: list[QueryResult] = []