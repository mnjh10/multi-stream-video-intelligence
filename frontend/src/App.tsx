import React, { useState, useEffect, useCallback } from 'react'
import { Header } from './components/Header'
import { Sidebar, type PageId } from './components/Sidebar'
import { EvidenceModal } from './components/EvidenceModal'
import { DashboardPage } from './pages/DashboardPage'
import { SearchPage } from './pages/SearchPage'
import { CamerasPage } from './pages/CamerasPage'
import { EventsPage } from './pages/EventsPage'
import { SystemPage } from './pages/SystemPage'
import { submitQuery } from './api/query'
import { requestEvidence } from './api/evidence'
import { fetchHealth } from './api/health'
import type { QueryResult, EvidenceResponse } from './types/api'

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<PageId>('dashboard')
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null)
  const [lastCheckedTime, setLastCheckedTime] = useState<string>('Connecting...')

  // Search state
  const [results, setResults] = useState<QueryResult[]>([])
  const [allSessionEvents, setAllSessionEvents] = useState<QueryResult[]>([])
  const [lastQuery, setLastQuery] = useState<string>('')
  const [isSearching, setIsSearching] = useState<boolean>(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [noMatchMessage, setNoMatchMessage] = useState<string | null>(null)

  // Temporal Follow-Up Context state
  const [activeContext, setActiveContext] = useState<{
    camera_id?: string
    object_id?: string
    timestamp?: number
    event_id?: string
    frame_index?: number
    bbox?: number[]
  } | null>(null)

  // Direct camera search navigation state
  const [selectedCameraFilter, setSelectedCameraFilter] = useState<string>('')

  // Evidence Modal state
  const [isEvidenceModalOpen, setIsEvidenceModalOpen] = useState<boolean>(false)
  const [activeEvidence, setActiveEvidence] = useState<EvidenceResponse | null>(null)
  const [isEvidenceLoading, setIsEvidenceLoading] = useState<boolean>(false)
  const [evidenceError, setEvidenceError] = useState<string | null>(null)

  // Poll backend health
  const checkHealthStatus = useCallback(async () => {
    try {
      const res = await fetchHealth()
      if (res.status === 'ok') {
        setBackendOnline(true)
        setLastCheckedTime(new Date().toLocaleTimeString())
      } else {
        setBackendOnline(false)
      }
    } catch {
      setBackendOnline(false)
      setLastCheckedTime(new Date().toLocaleTimeString())
    }
  }, [])

  useEffect(() => {
    checkHealthStatus()
    const interval = setInterval(checkHealthStatus, 10000)
    return () => clearInterval(interval)
  }, [checkHealthStatus])

  // Handle Query
  const handleSearch = useCallback(
    async (queryText: string, cameraFilter?: string, topK: number = 5) => {
      setIsSearching(true)
      setSearchError(null)
      setNoMatchMessage(null)
      setLastQuery(queryText)
      setSelectedCameraFilter(cameraFilter || '')

      try {
        const filters = cameraFilter ? { camera_id: cameraFilter } : undefined
        const response = await submitQuery({
          query: queryText,
          top_k: topK,
          filters: filters || null,
          context: activeContext || null,
        })

        if (response.status === 'ok') {
          setResults(response.results)
          setNoMatchMessage(null)
          if (response.results.length > 0) {
            const top = response.results[0]
            const ts = typeof top.timestamp === 'number' ? top.timestamp : top.best_timestamp || 0.0
            setActiveContext({
              camera_id: top.camera_id,
              object_id: top.object_id || undefined,
              timestamp: ts,
              event_id: top.event_id,
              frame_index: top.frame_index || undefined,
              bbox: top.bbox || undefined,
            })
          }
          // Merge into session events without duplicates
          setAllSessionEvents((prev) => {
            const map = new Map(prev.map((item) => [item.event_id, item]))
            response.results.forEach((item) => map.set(item.event_id, item))
            return Array.from(map.values())
          })
        } else if (response.status === 'no_match') {
          setResults([])
          setNoMatchMessage(response.message || 'No matching observations found.')
        } else {
          setSearchError(response.message || 'Search failed: Unexpected status from backend')
        }
      } catch (err: any) {
        setSearchError(err?.message || 'ARGUS backend is unavailable.')
      } finally {
        setIsSearching(false)
        setActivePage('search')
      }
    },
    [activeContext]
  )

  // Handle Evidence Extraction
  const handleViewEvidence = async (result: QueryResult) => {
    setIsEvidenceModalOpen(true)
    setIsEvidenceLoading(true)
    setEvidenceError(null)
    setActiveEvidence(null)

    // Also update activeContext to clicked item
    const ts =
      typeof result.timestamp === 'number'
        ? result.timestamp
        : typeof result.best_timestamp === 'number'
        ? result.best_timestamp
        : 0.0

    setActiveContext({
      camera_id: result.camera_id,
      object_id: result.object_id || undefined,
      timestamp: ts,
      event_id: result.event_id,
      frame_index: result.frame_index || undefined,
      bbox: result.bbox || undefined,
    })

    try {
      const videoPath =
        result.evidence?.source_video ||
        `data/videos/${result.camera_id}/vdo.avi`

      const res = await requestEvidence({
        source_video: videoPath,
        timestamp: ts,
        camera_id: result.camera_id,
        event_id: result.event_id,
        object_id: result.object_id || undefined,
        frame_index: result.frame_index || undefined,
        bbox: result.bbox || undefined,
        annotate: true,
      })

      setActiveEvidence(res)
    } catch (err: any) {
      setEvidenceError(err?.message || 'Evidence could not be generated.')
    } finally {
      setIsEvidenceLoading(false)
    }
  }

  // Handle Search from Camera Card
  const handleSearchCamera = (cameraId: string) => {
    setSelectedCameraFilter(cameraId)
    setActivePage('search')
    handleSearch(`vehicles in ${cameraId}`, cameraId, 5)
  }

  return (
    <div className="app-container">
      <Header backendOnline={backendOnline} activeQueryCount={results.length} />

      <div className="app-main">
        <Sidebar
          activePage={activePage}
          setActivePage={setActivePage}
          resultCount={allSessionEvents.length}
        />

        <main className="content-area">
          {activePage === 'dashboard' && (
            <DashboardPage
              onSearch={(q) => handleSearch(q, undefined, 5)}
              onNavigateToCameras={() => setActivePage('cameras')}
              recentResults={results}
              lastQuery={lastQuery}
              backendOnline={backendOnline}
              onViewEvidence={handleViewEvidence}
              isSearching={isSearching}
            />
          )}

          {activePage === 'search' && (
            <SearchPage
              onSearch={handleSearch}
              results={results}
              isLoading={isSearching}
              error={searchError}
              noMatchMessage={noMatchMessage}
              lastQuery={lastQuery}
              onViewEvidence={handleViewEvidence}
              initialQuery={lastQuery}
              initialCamera={selectedCameraFilter}
              activeContext={activeContext}
            />
          )}

          {activePage === 'cameras' && (
            <CamerasPage onSearchCamera={handleSearchCamera} />
          )}

          {activePage === 'events' && (
            <EventsPage
              events={allSessionEvents}
              onViewEvidence={handleViewEvidence}
              onNavigateToSearch={() => setActivePage('search')}
            />
          )}

          {activePage === 'system' && (
            <SystemPage
              backendOnline={backendOnline}
              lastCheckedTime={lastCheckedTime}
            />
          )}
        </main>
      </div>

      {isEvidenceModalOpen && (
        <EvidenceModal
          evidence={activeEvidence}
          loading={isEvidenceLoading}
          error={evidenceError}
          onClose={() => setIsEvidenceModalOpen(false)}
        />
      )}
    </div>
  )
}

export default App
