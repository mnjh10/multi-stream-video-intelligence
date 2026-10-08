from fastapi import APIRouter, HTTPException

from backend.schemas.evidence import EvidenceRequest, EvidenceResponse
from backend.services.evidence import EvidenceService

router = APIRouter(prefix="/evidence", tags=["evidence"])

evidence_service = EvidenceService()


@router.post("", response_model=EvidenceResponse)
def extract_evidence(request: EvidenceRequest):
    try:
        result = evidence_service.extract_frame_evidence(
            source_video=request.source_video,
            timestamp=request.timestamp,
            camera_id=request.camera_id,
            event_id=request.event_id,
            output_dir=request.output_dir,
        )
        return EvidenceResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
