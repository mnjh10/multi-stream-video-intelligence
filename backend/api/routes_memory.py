from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.schemas.memory import MemoryRequest, MemoryResponse
from backend.services.memory import MemoryService


router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("", response_model=MemoryResponse)
def store_memory(
    request: MemoryRequest,
    db: Session = Depends(get_db),
):
    service = MemoryService(db)

    memory = service.store_reference(
        key=request.key,
        value=request.value,
        camera_id=request.camera_id,
        region=request.region,
        metadata=request.metadata,
    )

    return MemoryResponse(
        memory_id=memory.memory_id,
        key=memory.key,
        value=memory.value,
        camera_id=memory.camera_id,
        region=memory.region,
        metadata=memory.metadata_,
    )


@router.get("/{key}", response_model=MemoryResponse | None)
def resolve_memory(
    key: str,
    db: Session = Depends(get_db),
):
    service = MemoryService(db)
    memory = service.resolve_reference(key)

    if memory is None:
        return None

    return MemoryResponse(
        memory_id=memory.memory_id,
        key=memory.key,
        value=memory.value,
        camera_id=memory.camera_id,
        region=memory.region,
        metadata=memory.metadata_,
    )