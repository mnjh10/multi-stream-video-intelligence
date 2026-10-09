import React from 'react'
import { Video, Search, CheckCircle2 } from 'lucide-react'
import { CITYFLOW_CAMERAS } from '../data/cameras'

import type { CameraData } from '../types/api'

interface CamerasPageProps {
  onSearchCamera: (cameraId: string) => void
}

export const CamerasPage: React.FC<CamerasPageProps> = ({ onSearchCamera }) => {
  return (
    <div>
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Video size={16} color="var(--accent-primary)" />
              CityFlowV2 Camera Streams Inventory
            </div>
            <div className="panel-description">
              Authoritative 11-camera multi-stream video dataset from AI City Challenge 2022
            </div>
          </div>
          <span className="badge badge-camera">11 STREAMS ACTIVE</span>
        </div>

        <div className="grid-3">
          {CITYFLOW_CAMERAS.map((cam: CameraData) => (
            <div
              key={cam.camera_id}
              style={{
                backgroundColor: 'var(--surface-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: 6,
                padding: 16,
                display: 'flex',
                flexDirection: 'column',
                gap: 12,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="badge badge-camera">{cam.camera_id}</span>
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
                  Available
                </span>
              </div>

              <div>
                <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
                  {cam.name}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 2 }}>
                  {cam.scenario}
                </div>
              </div>

              <div className="meta-grid">
                <div className="meta-item">
                  <span className="meta-label">Resolution</span>
                  <span className="meta-value">{cam.resolution}</span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Framerate</span>
                  <span className="meta-value">{cam.fps.toFixed(1)} FPS</span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Total Frames</span>
                  <span className="meta-value">{cam.total_frames.toLocaleString()}</span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Video Format</span>
                  <span className="meta-value">AVI (H.264)</span>
                </div>
              </div>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  paddingTop: 8,
                  borderTop: '1px solid var(--border-subtle)',
                }}
              >
                <span style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                  {cam.source_video}
                </span>

                <button
                  className="btn-secondary"
                  onClick={() => onSearchCamera(cam.camera_id)}
                  style={{ fontSize: 11 }}
                >
                  <Search size={12} />
                  <span>Search</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
