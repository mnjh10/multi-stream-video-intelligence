import React, { useState, useEffect } from 'react'
import { Search, Filter, AlertCircle, Sparkles, X } from 'lucide-react'
import { ResultCard } from '../components/ResultCard'
import type { QueryResult } from '../types/api'

interface SearchPageProps {
  onSearch: (query: string, cameraFilter?: string, topK?: number) => void
  results: QueryResult[]
  isLoading: boolean
  error: string | null
  noMatchMessage?: string | null
  lastQuery: string
  onViewEvidence: (result: QueryResult) => void
  initialQuery?: string
  initialCamera?: string
  activeContext?: {
    camera_id?: string
    object_id?: string
    timestamp?: number
    event_id?: string
  } | null
}

export const SearchPage: React.FC<SearchPageProps> = ({
  onSearch,
  results,
  isLoading,
  error,
  noMatchMessage,
  lastQuery,
  onViewEvidence,
  initialQuery = '',
  initialCamera = '',
  activeContext,
}) => {
  const [queryText, setQueryText] = useState(initialQuery)
  const [selectedCamera, setSelectedCamera] = useState(initialCamera)
  const [topK, setTopK] = useState(5)

  useEffect(() => {
    if (initialQuery !== undefined) {
      setQueryText(initialQuery)
    }
  }, [initialQuery])

  useEffect(() => {
    setSelectedCamera(initialCamera || '')
  }, [initialCamera])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (queryText.trim() && !isLoading) {
      onSearch(queryText.trim(), selectedCamera || undefined, topK)
    }
  }

  const suggestionQueries = [
    'Find a car in camera 1',
    'Find trucks in camera 11',
    'Find vehicles in camera 3',
    'white car on road',
    'red vehicle near camera 2',
  ]

  const camerasList = [
    'cam_01', 'cam_02', 'cam_03', 'cam_04', 'cam_05',
    'cam_06', 'cam_07', 'cam_08', 'cam_09', 'cam_10', 'cam_11',
  ]

  return (
    <div>
      {/* Search Header Panel */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Sparkles size={16} color="var(--accent-primary)" />
              Ask ARGUS — Natural Language Retrieval
            </div>
            <div className="panel-description">
              Describe vehicles, actions, or camera locations in plain conversational English
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="search-container">
          <div className="search-bar">
            <input
              type="text"
              className="search-input"
              placeholder="e.g. Find a car in camera 1, or white vehicle in intersection"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              disabled={isLoading}
            />
            {queryText && !isLoading && (
              <button
                type="button"
                onClick={() => setQueryText('')}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-dim)',
                  cursor: 'pointer',
                  padding: '4px 8px',
                  display: 'flex',
                  alignItems: 'center',
                }}
                title="Clear query"
              >
                <X size={14} />
              </button>
            )}
            <button
              type="submit"
              className="btn-primary"
              disabled={isLoading || !queryText.trim()}
            >
              <Search size={14} />
              <span>{isLoading ? 'Searching recorded video...' : 'Search'}</span>
            </button>
          </div>

          {/* Filters and Controls */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 12,
              marginTop: 4,
            }}
          >
            <div className="suggestions-row" style={{ flexWrap: 'wrap' }}>
              <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>Suggestions:</span>
              {activeContext && activeContext.object_id ? (
                <>
                  <button
                    type="button"
                    className="suggestion-pill"
                    style={{
                      borderColor: 'var(--accent-primary)',
                      color: 'var(--accent-primary)',
                      backgroundColor: 'rgba(0, 229, 255, 0.08)',
                    }}
                    onClick={() => {
                      const q = 'Where was this object 10 seconds earlier?'
                      setQueryText(q)
                      onSearch(q, undefined, topK)
                    }}
                  >
                    10s earlier
                  </button>
                  <button
                    type="button"
                    className="suggestion-pill"
                    style={{
                      borderColor: 'var(--accent-primary)',
                      color: 'var(--accent-primary)',
                      backgroundColor: 'rgba(0, 229, 255, 0.08)',
                    }}
                    onClick={() => {
                      const q = 'Show me this car later.'
                      setQueryText(q)
                      onSearch(q, undefined, topK)
                    }}
                  >
                    Show later
                  </button>
                  <button
                    type="button"
                    className="suggestion-pill"
                    style={{
                      borderColor: 'var(--accent-primary)',
                      color: 'var(--accent-primary)',
                      backgroundColor: 'rgba(0, 229, 255, 0.08)',
                    }}
                    onClick={() => {
                      const q = 'Show me this same car at its next appearance.'
                      setQueryText(q)
                      onSearch(q, undefined, topK)
                    }}
                  >
                    Next appearance
                  </button>
                </>
              ) : null}
              {suggestionQueries.map((q) => (
                <button
                  key={q}
                  type="button"
                  className="suggestion-pill"
                  onClick={() => {
                    setQueryText(q)
                    onSearch(q, selectedCamera || undefined, topK)
                  }}
                >
                  {q}
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)' }}>
                <Filter size={12} />
                <span>Camera:</span>
                <select
                  value={selectedCamera}
                  onChange={(e) => setSelectedCamera(e.target.value)}
                  style={{
                    backgroundColor: 'var(--surface-secondary)',
                    color: 'var(--text-primary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 4,
                    padding: '3px 8px',
                    fontSize: 12,
                    fontFamily: 'var(--font-mono)',
                    outline: 'none',
                  }}
                >
                  <option value="">All Cameras</option>
                  {camerasList.map((cam) => (
                    <option key={cam} value={cam}>
                      {cam}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)' }}>
                <span>Top K:</span>
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  style={{
                    backgroundColor: 'var(--surface-secondary)',
                    color: 'var(--text-primary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 4,
                    padding: '3px 8px',
                    fontSize: 12,
                    fontFamily: 'var(--font-mono)',
                    outline: 'none',
                  }}
                >
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                </select>
              </div>
            </div>
          </div>
        </form>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="panel">
          <div className="state-box">
            <div className="spinner" />
            <div style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-primary)' }}>
              Searching recorded video...
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>
              Querying OpenCLIP visual embeddings and executing temporal event clustering
            </div>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && !isLoading && (
        <div className="panel">
          <div className="state-box" style={{ color: 'var(--status-danger)' }}>
            <AlertCircle size={28} />
            <div style={{ fontSize: 14, fontWeight: 600 }}>ARGUS backend is unavailable.</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{error}</div>
          </div>
        </div>
      )}

      {/* Results State */}
      {!isLoading && !error && results.length > 0 && (
        <div className="panel">
          <div className="panel-header">
            <div>
              <div className="panel-title">
                Retrieved Events ({results.length})
              </div>
              <div className="panel-description">
                Showing temporal event clusters ranked by semantic query alignment
              </div>
            </div>
            <span
              style={{
                fontSize: 11,
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-dim)',
              }}
            >
              Query: "{lastQuery}"
            </span>
          </div>

          <div className="grid-3">
            {results.map((item) => (
              <ResultCard
                key={item.event_id}
                result={item}
                onViewEvidence={onViewEvidence}
              />
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && !error && results.length === 0 && (lastQuery || noMatchMessage) && (
        <div className="panel">
          <div className="state-box">
            <div style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-muted)' }}>
              No matching events found.
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', maxWidth: 500, textAlign: 'center' }}>
              {noMatchMessage || 'Try broadening your query or selecting a different camera filter.'}
            </div>
          </div>
        </div>
      )}

      {/* Idle Guidance State */}
      {!isLoading && !error && results.length === 0 && !lastQuery && (
        <div className="panel">
          <div className="state-box">
            <Search size={32} color="var(--border-color)" />
            <div style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-muted)' }}>
              Ready for Conversational Search
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', maxWidth: 440 }}>
              Type an inquiry above such as <em>"Find a car in camera 1"</em> to search through indexed CityFlowV2 multi-camera observations.
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
