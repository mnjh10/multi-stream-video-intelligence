from fastapi import APIRouter

from backend.schemas.evidence import EvidenceRequest, EvidenceResponse
from backend.services.evidence import EvidenceService


router = APIRouter(prefix="/evidence", tags=["evidence"])

evidence_service = EvidenceService()


@router.post("", response_model=EvidenceResponse)
def generate_evidence(request: EvidenceRequest):
    result = evidence_service.generate_evidence(
        source_video=request.source_video,
        timestamp=request.timestamp,
        event_id=request.event_id,
    )

    return EvidenceResponse(**result)