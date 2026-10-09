import React from 'react'
import { LayoutDashboard, Search, Video, Film, Cpu } from 'lucide-react'

export type PageId = 'dashboard' | 'search' | 'cameras' | 'events' | 'system'

interface SidebarProps {
  activePage: PageId
  setActivePage: (page: PageId) => void
  resultCount?: number
}

export const Sidebar: React.FC<SidebarProps> = ({ activePage, setActivePage, resultCount = 0 }) => {
  const navItems: { id: PageId; label: string; icon: React.ReactNode; badge?: number }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={16} /> },
    { id: 'search', label: 'Search', icon: <Search size={16} /> },
    { id: 'cameras', label: 'Cameras', icon: <Video size={16} /> },
    { id: 'events', label: 'Events', icon: <Film size={16} />, badge: resultCount > 0 ? resultCount : undefined },
    { id: 'system', label: 'System', icon: <Cpu size={16} /> },
  ]

  return (
    <aside className="sidebar">
      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setActivePage(item.id)}
            className={`nav-item ${activePage === item.id ? 'active' : ''}`}
          >
            {item.icon}
            <span className="nav-text" style={{ flex: 1 }}>{item.label}</span>
            {item.badge !== undefined && (
              <span
                style={{
                  fontSize: 10,
                  fontFamily: 'var(--font-mono)',
                  background: 'var(--surface-secondary)',
                  padding: '2px 6px',
                  borderRadius: 4,
                  border: '1px solid var(--border-color)',
                }}
              >
                {item.badge}
              </span>
            )}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div>ARGUS Console v1.0</div>
        <div style={{ marginTop: 4, color: 'var(--text-dim)' }}>CityFlowV2 Pipeline</div>
      </div>
    </aside>
  )
}
