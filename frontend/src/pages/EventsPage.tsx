import React, { useState } from 'react'
import { Film, Filter, Search } from 'lucide-react'
import { ResultCard } from '../components/ResultCard'
import type { QueryResult } from '../types/api'

interface EventsPageProps {
  events: QueryResult[]
  onViewEvidence: (result: QueryResult) => void
  onNavigateToSearch: () => void
}

export const EventsPage: React.FC<EventsPageProps> = ({
  events,
  onViewEvidence,
  onNavigateToSearch,
}) => {
  const [filterCamera, setFilterCamera] = useState('')
  const [filterObjectType, setFilterObjectType] = useState('')

  const filteredEvents = events.filter((ev) => {
    if (filterCamera && ev.camera_id !== filterCamera) return false
    if (filterObjectType && ev.object_type !== filterObjectType) return false
    return true
  })

  const uniqueCameras = Array.from(new Set(events.map((e) => e.camera_id))).sort()
  const uniqueObjectTypes = Array.from(
    new Set(events.map((e) => e.object_type).filter(Boolean))
  ).sort()

  return (
    <div>
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Film size={16} color="var(--accent-primary)" />
              Retrieved Events Explorer
            </div>
            <div className="panel-description">
              Browse and inspect all temporal events retrieved during the active intelligence session
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span
              style={{
                fontSize: 12,
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
              }}
            >
              Total: {events.length} Events
            </span>
            <button className="btn-secondary" onClick={onNavigateToSearch}>
              <Search size={13} />
              <span>New Search</span>
            </button>
          </div>
        </div>

        {events.length > 0 && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 16,
              marginBottom: 16,
              padding: '10px 14px',
              backgroundColor: 'var(--surface-secondary)',
              borderRadius: 6,
              border: '1px solid var(--border-color)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)' }}>
              <Filter size={13} />
              <span>Filter by Camera:</span>
              <select
                value={filterCamera}
                onChange={(e) => setFilterCamera(e.target.value)}
                style={{
                  backgroundColor: 'var(--surface-primary)',
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
                {uniqueCameras.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)' }}>
              <span>Filter by Object:</span>
              <select
                value={filterObjectType}
                onChange={(e) => setFilterObjectType(e.target.value)}
                style={{
                  backgroundColor: 'var(--surface-primary)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 4,
                  padding: '3px 8px',
                  fontSize: 12,
                  fontFamily: 'var(--font-mono)',
                  outline: 'none',
                }}
              >
                <option value="">All Objects</option>
                {uniqueObjectTypes.map((o) => (
                  <option key={o as string} value={o as string}>
                    {o}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}

        {filteredEvents.length > 0 ? (
          <div className="grid-3">
            {filteredEvents.map((item) => (
              <ResultCard
                key={item.event_id}
                result={item}
                onViewEvidence={onViewEvidence}
              />
            ))}
          </div>
        ) : events.length === 0 ? (
          <div className="state-box">
            <Film size={32} color="var(--border-color)" />
            <div style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-muted)' }}>
              No Events Retrieved Yet
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', maxWidth: 420 }}>
              Perform a natural-language search in the Search tab to populate real temporal events from the recorded CityFlowV2 multi-camera streams.
            </div>
            <button className="btn-primary" onClick={onNavigateToSearch} style={{ marginTop: 8 }}>
              <Search size={14} />
              <span>Go to Conversational Search</span>
            </button>
          </div>
        ) : (
          <div className="state-box">
            <div style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-muted)' }}>
              No events match the active filters.
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
