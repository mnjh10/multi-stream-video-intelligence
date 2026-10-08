from pydantic import BaseModel


class CameraCreate(BaseModel):
    camera_id: str
    name: str
    source: str
    metadata: dict = {}


class CameraResponse(BaseModel):
    camera_id: str
    name: str
    source: str
    metadata: dict