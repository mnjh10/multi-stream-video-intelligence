import React, { useState } from 'react'
import { Search, Video, Cpu, Clock } from 'lucide-react'
import { CITYFLOW_CAMERAS } from '../data/cameras'

import { ResultCard } from '../components/ResultCard'
import type { QueryResult } from '../types/api'

interface DashboardPageProps {
  onSearch: (query: string) => void
  onNavigateToCameras: () => void
  recentResults: QueryResult[]
  lastQuery: string
  backendOnline: boolean | null
  onViewEvidence: (result: QueryResult) => void
  isSearching: boolean
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  onSearch,
  onNavigateToCameras,
  recentResults,
  lastQuery,
  backendOnline,
  onViewEvidence,
  isSearching,
}) => {
  const [inputVal, setInputVal] = useState('')

  const handleQuickSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (inputVal.trim()) {
      onSearch(inputVal.trim())
    }
  }

  const exampleQueries = [
    'Find a car in camera 1',
    'Find trucks in camera 11',
    'Find vehicles in camera 3',
    'white car',
  ]

  return (
    <div>
      {/* Overview Status Panel */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Cpu size={16} color="var(--accent-primary)" />
              System Status Overview
            </div>
            <div className="panel-description">
              Multi-camera intelligence operational state and services
            </div>
          </div>
          <span
            className="badge"
            style={{
              backgroundColor: backendOnline ? 'var(--status-success-bg)' : 'var(--status-danger-bg)',
              color: backendOnline ? 'var(--status-success)' : 'var(--status-danger)',
              border: `1px solid ${backendOnline ? 'var(--status-success)' : 'var(--status-danger)'}`,
            }}
          >
            {backendOnline ? 'ALL SYSTEMS NOMINAL' : 'BACKEND OFFLINE'}
          </span>
        </div>

        <div className="grid-3">
          <div
            style={{
              backgroundColor: 'var(--surface-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: 6,
              padding: 14,
            }}
          >
            <div style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
              Backend API
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, marginTop: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className={`status-dot ${backendOnline ? 'online' : 'offline'}`} />
              {backendOnline ? 'FastAPI Operational' : 'Connection Refused'}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
              Routes: /query, /evidence, /health, /memory
            </div>
          </div>

          <div
            style={{
              backgroundColor: 'var(--surface-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: 6,
              padding: 14,
            }}
          >
            <div style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
              Retrieval Engine
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, marginTop: 4, color: 'var(--accent-primary)' }}>
              OpenCLIP ViT-B-32 + FAISS
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
              Demonstration index: 100 observations
            </div>
          </div>

          <div
            style={{
              backgroundColor: 'var(--surface-secondary)',
              border: '1px solid var(--border-color)',
              borderRadius: 6,
              padding: 14,
            }}
          >
            <div style={{ fontSize: 11, color: 'var(--text-dim)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
              Evidence Service
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, marginTop: 4, color: 'var(--status-success)' }}>
              OpenCV Frame Extractor
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
              Native video playback frame seek
            </div>
          </div>
        </div>
      </div>

      {/* Quick Search Console */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Search size={16} color="var(--accent-primary)" />
              Conversational Query Console
            </div>
            <div className="panel-description">
              Natural-language search across recorded video streams
            </div>
          </div>
        </div>

        <form onSubmit={handleQuickSubmit} className="search-container">
          <div className="search-bar">
            <input
              type="text"
              className="search-input"
              placeholder="Ask ARGUS: e.g. Find a car in camera 1"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              disabled={isSearching}
            />
            <button type="submit" className="btn-primary" disabled={isSearching || !inputVal.trim()}>
              <Search size={14} />
              <span>{isSearching ? 'Searching...' : 'Search'}</span>
            </button>
          </div>

          <div className="suggestions-row">
            <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>Suggestions:</span>
            {exampleQueries.map((ex) => (
              <button
                key={ex}
                type="button"
                className="suggestion-pill"
                onClick={() => {
                  setInputVal(ex)
                  onSearch(ex)
                }}
              >
                {ex}
              </button>
            ))}
          </div>
        </form>
      </div>

      {/* Camera Overview Snippet */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Video size={16} color="var(--accent-primary)" />
              Recorded Video Streams ({CITYFLOW_CAMERAS.length} Cameras)
            </div>
            <div className="panel-description">
              Authoritative CityFlowV2 multi-camera streams
            </div>
          </div>
          <button className="btn-secondary" onClick={onNavigateToCameras}>
            View All Cameras
          </button>
        </div>

        <div className="grid-4">
          {CITYFLOW_CAMERAS.slice(0, 4).map((cam) => (
            <div
              key={cam.camera_id}
              style={{
                backgroundColor: 'var(--surface-secondary)',
                border: '1px solid var(--border-color)',
                borderRadius: 6,
                padding: 12,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span className="badge badge-camera">{cam.camera_id}</span>
                <span style={{ fontSize: 10, color: 'var(--status-success)', fontFamily: 'var(--font-mono)' }}>
                  {cam.fps} FPS
                </span>
              </div>
              <div style={{ fontSize: 13, fontWeight: 500, marginTop: 6, color: 'var(--text-primary)' }}>
                {cam.name}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 2 }}>
                {cam.scenario} • {cam.resolution}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Query Results Area */}
      {recentResults.length > 0 && (
        <div className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <Clock size={16} color="var(--accent-primary)" />
                Latest Retrieval Results ({recentResults.length} Events)
              </div>
              <div className="panel-description">
                Query: "{lastQuery}"
              </div>
            </div>
          </div>

          <div className="grid-3">
            {recentResults.map((res) => (
              <ResultCard
                key={res.event_id}
                result={res}
                onViewEvidence={onViewEvidence}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
