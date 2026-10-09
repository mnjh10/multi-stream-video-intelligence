import React from 'react'
import { Eye, Clock, Film } from 'lucide-react'
import type { QueryResult } from '../types/api'


interface ResultCardProps {
  result: QueryResult
  onViewEvidence: (result: QueryResult) => void
  isEvidenceLoading?: boolean
}

export const ResultCard: React.FC<ResultCardProps> = ({
  result,
  onViewEvidence,
  isEvidenceLoading = false,
}) => {
  const formattedTimestamp =
    typeof result.timestamp === 'number'
      ? `${result.timestamp.toFixed(2)}s`
      : result.timestamp
      ? `${result.timestamp}`
      : '0.00s'

  const scoreFormatted =
    typeof result.score === 'number'
      ? result.score.toFixed(3)
      : 'N/A'

  return (
    <div className="result-card">
      <div className="card-top">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <span className="badge badge-camera">{result.camera_id}</span>
          {result.object_type && (
            <span className="badge badge-object">{result.object_type}</span>
          )}
          {result.verification_status === 'attribute_verified' && (
            <span
              className="badge"
              style={{
                backgroundColor: 'rgba(0, 230, 118, 0.15)',
                color: '#00e676',
                border: '1px solid rgba(0, 230, 118, 0.4)',
                fontWeight: 600,
              }}
            >
              ✓ Verified{result.attribute_details?.colors ? `: ${result.attribute_details.colors.join(', ')}` : ''}
            </span>
          )}
          {result.verification_status === 'visual_similarity' && (
            <span
              className="badge"
              style={{
                backgroundColor: 'rgba(255, 255, 255, 0.08)',
                color: 'var(--text-secondary)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              Visual Similarity
            </span>
          )}
          {result.is_followup && (
            <span
              className="badge"
              style={{
                backgroundColor: 'rgba(0, 229, 255, 0.15)',
                color: 'var(--accent-primary)',
                border: '1px solid rgba(0, 229, 255, 0.4)',
              }}
            >
              Follow-Up{result.followup_relation ? `: ${result.followup_relation}` : ''}
            </span>
          )}
        </div>
        <span className="badge badge-score">Score: {scoreFormatted}</span>
      </div>

      <div className="meta-grid">
        <div className="meta-item">
          <span className="meta-label">Event ID</span>
          <span className="meta-value" style={{ color: 'var(--accent-primary)' }}>
            {result.event_id}
          </span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Timestamp</span>
          <span className="meta-value" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Clock size={12} color="var(--text-muted)" />
            {formattedTimestamp}
          </span>
        </div>
        {result.object_id && (
          <div className="meta-item">
            <span className="meta-label">Object ID</span>
            <span className="meta-value">{result.object_id}</span>
          </div>
        )}
        {result.observation_id && (
          <div className="meta-item">
            <span className="meta-label">Observation ID</span>
            <span className="meta-value" style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              {result.observation_id}
            </span>
          </div>
        )}
        {typeof result.frame_index === 'number' && (
          <div className="meta-item">
            <span className="meta-label">Frame Index</span>
            <span className="meta-value">#{result.frame_index}</span>
          </div>
        )}
        {typeof result.timestamp_start === 'number' && typeof result.timestamp_end === 'number' && !result.is_followup && (
          <div className="meta-item">
            <span className="meta-label">Event Window</span>
            <span className="meta-value">
              {result.timestamp_start.toFixed(1)}s - {result.timestamp_end.toFixed(1)}s
            </span>
          </div>
        )}
        {result.bbox && result.bbox.length === 4 && (
          <div className="meta-item">
            <span className="meta-label">Ground BBox</span>
            <span className="meta-value" style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}>
              [{result.bbox.map((v) => Math.round(v)).join(', ')}]
            </span>
          </div>
        )}
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginTop: 4,
          paddingTop: 8,
          borderTop: '1px solid var(--border-subtle)',
        }}
      >
        <span
          style={{
            fontSize: 11,
            color: 'var(--text-dim)',
            fontFamily: 'var(--font-mono)',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <Film size={12} />
          {result.evidence?.source_video ? 'Source Video Linked' : 'Ground-Truth Match'}
        </span>

        <button
          className="btn-secondary"
          onClick={() => onViewEvidence(result)}
          disabled={isEvidenceLoading}
          style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}
        >
          <Eye size={13} />
          <span>{isEvidenceLoading ? 'Extracting...' : 'View Evidence'}</span>
        </button>
      </div>
    </div>
  )
}
