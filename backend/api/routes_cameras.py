from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db.models import Camera
from backend.db.session import get_db
from backend.schemas.camera import CameraCreate, CameraResponse


router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.post("", response_model=CameraResponse)
def create_camera(
    request: CameraCreate,
    db: Session = Depends(get_db),
):
    existing = db.scalar(
        select(Camera).where(Camera.camera_id == request.camera_id)
    )

    if existing:
        raise HTTPException(
            status_code=409,
            detail="Camera already exists",
        )

    camera = Camera(
        camera_id=request.camera_id,
        name=request.name,
        source=request.source,
        metadata_=request.metadata,
    )

    db.add(camera)
    db.commit()
    db.refresh(camera)

    return CameraResponse(
        camera_id=camera.camera_id,
        name=camera.name,
        source=camera.source,
        metadata=camera.metadata_,
    )


@router.get("", response_model=list[CameraResponse])
def list_cameras(
    db: Session = Depends(get_db),
):
    cameras = db.scalars(
        select(Camera)
    ).all()

    return [
        CameraResponse(
            camera_id=camera.camera_id,
            name=camera.name,
            source=camera.source,
            metadata=camera.metadata_,
        )
        for camera in cameras
    ]


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(
    camera_id: str,
    db: Session = Depends(get_db),
):
    camera = db.scalar(
        select(Camera).where(Camera.camera_id == camera_id)
    )

    if camera is None:
        raise HTTPException(
            status_code=404,
            detail="Camera not found",
        )

    return CameraResponse(
        camera_id=camera.camera_id,
        name=camera.name,
        source=camera.source,
        metadata=camera.metadata_,
    )