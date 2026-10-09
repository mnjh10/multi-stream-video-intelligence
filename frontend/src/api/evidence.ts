import { apiFetch } from './client'
import type { EvidenceRequest, EvidenceResponse } from '../types/api'

export async function requestEvidence(payload: EvidenceRequest): Promise<EvidenceResponse> {
  return apiFetch<EvidenceResponse>('/evidence', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}
