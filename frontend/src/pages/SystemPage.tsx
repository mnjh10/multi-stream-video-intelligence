import React from 'react'
import { Cpu, CheckCircle2, AlertCircle, Database, ShieldCheck, Layers, Terminal } from 'lucide-react'

interface SystemPageProps {
  backendOnline: boolean | null
  lastCheckedTime?: string
}

export const SystemPage: React.FC<SystemPageProps> = ({
  backendOnline,
  lastCheckedTime = 'Just now',
}) => {
  return (
    <div>
      {/* System Status Panel */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Cpu size={16} color="var(--accent-primary)" />
              ARGUS System & Architecture Diagnostics
            </div>
            <div className="panel-description">
              Operational health of backend APIs, retrieval engine, and video processing layers
            </div>
          </div>
          <span style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
            Polled: {lastCheckedTime}
          </span>
        </div>

        <div className="grid-3">
          {/* Backend Status */}
          <div
            style={{
              backgroundColor: 'var(--surface-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: 6,
              padding: 16,
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                Application Backend
              </span>
              <span
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  fontSize: 11,
                  color: backendOnline ? 'var(--status-success)' : 'var(--status-danger)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {backendOnline ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
                {backendOnline ? 'Online' : 'Offline'}
              </span>
            </div>

            <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>
              FastAPI REST Server
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Bound to port 8000. Provides structured query execution, semantic memory resolution, and evidence dispatch.
            </div>
            <div
              style={{
                marginTop: 'auto',
                paddingTop: 8,
                borderTop: '1px solid var(--border-subtle)',
                fontSize: 11,
                fontFamily: 'var(--font-mono)',
                color: 'var(--accent-primary)',
              }}
            >
              Endpoints: /query • /evidence • /health • /memory
            </div>
          </div>

          {/* Retrieval Engine */}
          <div
            style={{
              backgroundColor: 'var(--surface-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: 6,
              padding: 16,
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                Vector Search Layer
              </span>
              <span
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  fontSize: 11,
                  color: 'var(--status-success)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                <CheckCircle2 size={12} />
                Real Provider
              </span>
            </div>

            <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>
              OpenCLIP + FAISS IndexFlatIP
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Encapsulated inside RealRetrievalProvider. Query parsing, metadata filtering, cosine similarity ranking, and temporal event grouping.
            </div>
            <div
              style={{
                marginTop: 'auto',
                paddingTop: 8,
                borderTop: '1px solid var(--border-subtle)',
                fontSize: 11,
                fontFamily: 'var(--font-mono)',
                color: 'var(--status-warning)',
              }}
            >
              Demonstration Index: 100 Observations
            </div>
          </div>

          {/* Evidence Service */}
          <div
            style={{
              backgroundColor: 'var(--surface-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: 6,
              padding: 16,
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                Ground-Truth Extractor
              </span>
              <span
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  fontSize: 11,
                  color: 'var(--status-success)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                <CheckCircle2 size={12} />
                OpenCV Active
              </span>
            </div>

            <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>
              Native Video Evidence Service
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Directly seeks source CityFlowV2 videos at designated float seconds to extract full 1080p evidence frames.
            </div>
            <div
              style={{
                marginTop: 'auto',
                paddingTop: 8,
                borderTop: '1px solid var(--border-subtle)',
                fontSize: 11,
                fontFamily: 'var(--font-mono)',
                color: 'var(--status-success)',
              }}
            >
              Target Directory: data/evidence/
            </div>
          </div>
        </div>
      </div>

      {/* Dataset & Index Truth Disclosure Panel */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Database size={16} color="var(--accent-primary)" />
              Dataset & Index Truth Disclosure
            </div>
            <div className="panel-description">
              Explicit distinction between total frozen pipeline corpus and active demonstration index
            </div>
          </div>
        </div>

        <div className="grid-2">
          <div
            style={{
              backgroundColor: 'var(--surface-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: 6,
              padding: 16,
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--status-warning)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <Layers size={15} />
              Active Demonstration Index (FAISS)
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, lineHeight: 1.6 }}>
              The currently persisted vector index (<code style={{ color: 'var(--accent-primary)', fontFamily: 'var(--font-mono)' }}>data/index/argus.index</code>) holds <strong>100 observations</strong> representing a balanced multi-camera slice across cameras 1 through 5. This ensures instant cold startup and sub-second retrieval latency during testing and evaluation.
            </div>
            <div className="meta-grid" style={{ marginTop: 12 }}>
              <div className="meta-item">
                <span className="meta-label">Indexed Count</span>
                <span className="meta-value" style={{ color: 'var(--status-warning)' }}>100 Observations</span>
              </div>
              <div className="meta-item">
                <span className="meta-label">Vector Dimension</span>
                <span className="meta-value">512 (ViT-B-32)</span>
              </div>
            </div>
          </div>

          <div
            style={{
              backgroundColor: 'var(--surface-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: 6,
              padding: 16,
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <ShieldCheck size={15} />
              Frozen Computer Vision Pipeline (Corpus)
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 8, lineHeight: 1.6 }}>
              Person 1's upstream computer vision pipeline is 100% frozen and validated across the entire CityFlowV2 dataset. Full-scale indexing can be executed offline through the provided index builder at any time.
            </div>
            <div className="meta-grid" style={{ marginTop: 12 }}>
              <div className="meta-item">
                <span className="meta-label">Total Observations</span>
                <span className="meta-value">61,039</span>
              </div>
              <div className="meta-item">
                <span className="meta-label">Total Crops</span>
                <span className="meta-value">61,039</span>
              </div>
              <div className="meta-item">
                <span className="meta-label">Sampled Frames</span>
                <span className="meta-value">7,287</span>
              </div>
              <div className="meta-item">
                <span className="meta-label">YOLO11 Detections</span>
                <span className="meta-value">68,906</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Architectural Guarantee Panel */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Terminal size={16} color="var(--accent-primary)" />
              Architectural Pipeline Flow
            </div>
            <div className="panel-description">
              Strict tier-separated video intelligence pipeline
            </div>
          </div>
        </div>

        <div
          style={{
            backgroundColor: 'var(--surface-secondary)',
            borderRadius: 6,
            border: '1px solid var(--border-color)',
            padding: 14,
            fontFamily: 'var(--font-mono)',
            fontSize: 12,
            color: 'var(--text-muted)',
            lineHeight: 1.8,
            overflowX: 'auto',
          }}
        >
          <span style={{ color: 'var(--accent-primary)' }}>Frontend UI (React/Vite)</span>
          {' → '}
          <span style={{ color: 'var(--text-primary)' }}>POST /query (FastAPI)</span>
          {' → '}
          <span style={{ color: 'var(--status-success)' }}>RealRetrievalProvider</span>
          {' → '}
          <span style={{ color: 'var(--text-primary)' }}>ObservationRetrievalPipeline</span>
          {' → '}
          <span style={{ color: 'var(--accent-primary)' }}>OpenCLIP + FAISS Index</span>
          {' → '}
          <span style={{ color: 'var(--status-warning)' }}>Temporal Grouping</span>
          {' → '}
          <span style={{ color: 'var(--status-success)' }}>POST /evidence (OpenCV Frame)</span>
        </div>
      </div>
    </div>
  )
}
