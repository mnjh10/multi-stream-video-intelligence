import { apiFetch } from './client'
import type { HealthResponse } from '../types/api'

export async function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/health', {
    method: 'GET',
  })
}
