export interface EvidenceMetadata {
  frame_path?: string | null
  clip_path?: string | null
  source_video?: string | null
}

export interface QueryResult {
  event_id: string
  camera_id: string
  timestamp: number | string
  best_timestamp?: number
  timestamp_start?: number
  timestamp_end?: number
  score: number | null
  object_id?: string | null
  object_type?: string | null
  evidence?: EvidenceMetadata | null
  bbox?: number[] | null
  frame_index?: number | null
  crop_path?: string | null
  observation_id?: string | null
  verification_status?: string | null
  attribute_details?: Record<string, any> | null
  is_followup?: boolean
  followup_relation?: string | null
}

export interface QueryRequest {
  query: string
  top_k?: number
  filters?: Record<string, any> | null
  context?: Record<string, any> | null
}

export interface QueryResponse {
  query: string
  status: string
  message?: string | null
  results: QueryResult[]
}

export interface EvidenceRequest {
  source_video: string
  timestamp: number
  camera_id?: string | null
  event_id?: string | null
  object_id?: string | null
  frame_index?: number | null
  bbox?: number[] | null
  annotate?: boolean
  output_dir?: string | null
}

export interface EvidenceExtractionMetadata {
  fps?: number
  height?: number
  width?: number
  total_frames?: number
}

export interface EvidenceResponse {
  status: string
  frame_path: string
  timestamp: number
  source_video: string
  camera_id?: string | null
  event_id?: string | null
  object_id?: string | null
  frame_index?: number | null
  bbox?: number[] | null
  annotated?: boolean
  raw_frame_path?: string | null
  message?: string | null
  metadata?: EvidenceExtractionMetadata | null
}

export interface HealthResponse {
  status: string
}

export interface CameraData {
  camera_id: string
  scenario: string
  name: string
  fps: number
  resolution: string
  total_frames: number
  source_video: string
}
