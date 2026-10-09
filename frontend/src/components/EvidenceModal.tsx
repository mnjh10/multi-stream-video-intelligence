import React from 'react'
import { X, Film, CheckCircle2, AlertCircle } from 'lucide-react'
import type { EvidenceResponse } from '../types/api'

interface EvidenceModalProps {
  evidence: EvidenceResponse | null
  loading: boolean
  error: string | null
  onClose: () => void
}

export const EvidenceModal: React.FC<EvidenceModalProps> = ({
  evidence,
  loading,
  error,
  onClose,
}) => {
  const [showRaw, setShowRaw] = React.useState(false)

  React.useEffect(() => {
    setShowRaw(false)
  }, [evidence])

  if (!evidence && !loading && !error) return null

  const activePath = showRaw && evidence?.raw_frame_path ? evidence.raw_frame_path : evidence?.frame_path
  const imageSrc = activePath
    ? activePath.startsWith('/')
      ? activePath
      : `/${activePath}`
    : ''

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <Film size={18} color="var(--accent-primary)" />
            <h3 style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>
              Evidence Frame Inspector
            </h3>
            {evidence?.camera_id && (
              <span className="badge badge-camera">{evidence.camera_id}</span>
            )}
            {evidence?.object_id && (
              <span className="badge badge-object">{evidence.object_id}</span>
            )}
            {evidence?.annotated && (
              <span
                className="badge"
                style={{
                  backgroundColor: 'rgba(0, 229, 255, 0.15)',
                  color: 'var(--accent-primary)',
                  border: '1px solid rgba(0, 229, 255, 0.4)',
                }}
              >
                Annotated BBox
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              display: 'flex',
              padding: 4,
            }}
          >
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {loading && (
            <div className="state-box">
              <div className="spinner" />
              <div>Generating evidence from recorded CityFlowV2 video...</div>
              <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                Seeking exact source video timestamp via OpenCV
              </div>
            </div>
          )}

          {error && (
            <div className="state-box" style={{ color: 'var(--status-danger)' }}>
              <AlertCircle size={24} />
              <div>Evidence could not be generated.</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{error}</div>
            </div>
          )}

          {evidence && !loading && (
            <>
              <div className="evidence-img-container">
                <img
                  src={imageSrc}
                  alt={`Evidence frame at ${evidence.timestamp}s`}
                  className="evidence-img"
                  onError={(e) => {
                    // Fallback visual indicator if image fails to render
                    e.currentTarget.style.display = 'none'
                  }}
                />
              </div>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 8,
                  flexWrap: 'wrap',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    fontSize: 12,
                    color: evidence.annotated ? 'var(--status-success)' : 'var(--text-muted)',
                  }}
                >
                  {evidence.annotated ? (
                    <CheckCircle2 size={14} color="var(--status-success)" />
                  ) : (
                    <AlertCircle size={14} color="var(--text-muted)" />
                  )}
                  <span>
                    {evidence.message
                      ? evidence.message
                      : evidence.annotated
                      ? 'Target bounding box localized and drawn with OpenCV'
                      : 'Extracted ground-truth frame grounded in source video playback'}
                  </span>
                </div>

                {evidence.annotated && evidence.raw_frame_path && (
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ fontSize: 11, padding: '4px 10px' }}
                    onClick={() => setShowRaw(!showRaw)}
                  >
                    {showRaw ? 'Show Bounding Box' : 'Show Raw Frame'}
                  </button>
                )}
              </div>

              <div className="meta-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
                <div className="meta-item">
                  <span className="meta-label">Camera</span>
                  <span className="meta-value">{evidence.camera_id || 'N/A'}</span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Timestamp</span>
                  <span className="meta-value">{evidence.timestamp.toFixed(2)}s</span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Frame Index</span>
                  <span className="meta-value">{evidence.frame_index ?? 'N/A'}</span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">FPS</span>
                  <span className="meta-value">{evidence.metadata?.fps ?? 10.0}</span>
                </div>
              </div>

              <div className="meta-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
                <div className="meta-item">
                  <span className="meta-label">Frame Dimensions</span>
                  <span className="meta-value">
                    {evidence.metadata?.width && evidence.metadata?.height
                      ? `${evidence.metadata.width} x ${evidence.metadata.height}`
                      : '1920 x 1080'}
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">
                    {evidence.bbox ? 'Detected BBox' : 'Source Video'}
                  </span>
                  <span className="meta-value" style={{ wordBreak: 'break-all', fontFamily: evidence.bbox ? 'var(--font-mono)' : undefined }}>
                    {evidence.bbox
                      ? `[${evidence.bbox.map((v) => Math.round(v)).join(', ')}]`
                      : evidence.source_video}
                  </span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
