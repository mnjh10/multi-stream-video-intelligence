import React from 'react'

interface HeaderProps {
  backendOnline: boolean | null
  activeQueryCount?: number
}

export const Header: React.FC<HeaderProps> = ({ backendOnline }) => {
  return (
    <header className="header-bar">
      <div className="header-left">
        <div className="brand-badge">
          <span className="brand-title">ARGUS</span>
          <span className="brand-subtitle">Conversational Multi-Camera Video Intelligence</span>
        </div>
      </div>
      <div className="header-right">
        <div className="status-pill">
          <span className="status-dot info" />
          <span>Recorded Analysis</span>
        </div>

        <div className="status-pill">
          <span className={`status-dot ${backendOnline === true ? 'online' : backendOnline === false ? 'offline' : 'info'}`} />
          <span>
            Backend: {backendOnline === true ? 'Online' : backendOnline === false ? 'Offline' : 'Checking...'}
          </span>
        </div>

        <div className="status-pill">
          <span className="status-dot online" />
          <span>FAISS: 100 Obs</span>
        </div>
      </div>
    </header>
  )
}
