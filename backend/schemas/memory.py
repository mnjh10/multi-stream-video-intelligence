from pydantic import BaseModel


class MemoryRequest(BaseModel):
    key: str
    value: str
    camera_id: str | None = None
    region: str | None = None
    metadata: dict | None = None


class MemoryResponse(BaseModel):
    memory_id: int
    key: str
    value: str
    camera_id: str | None = None
    region: str | None = None
    metadata: dict