import { apiFetch } from './client'
import type { QueryRequest, QueryResponse } from '../types/api'

export async function submitQuery(payload: QueryRequest): Promise<QueryResponse> {
  return apiFetch<QueryResponse>('/query', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}
