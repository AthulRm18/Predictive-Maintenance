import { useState } from 'react'
import './index.css'
import OverviewPage from './pages/Overview'
import FleetPage from './pages/Fleet'
import ReviewPage from './pages/Review'
import WhatIfPage from './pages/WhatIf'

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: '📊' },
  { id: 'fleet', label: 'Fleet Map', icon: '🏭' },
  { id: 'review', label: 'Review Queue', icon: '👁' },
  { id: 'whatif', label: 'What-If', icon: '🔮' },
]

function App() {
  const [activePage, setActivePage] = useState('overview')

  const renderPage = () => {
    switch (activePage) {
      case 'overview': return <OverviewPage />
      case 'fleet': return <FleetPage />
      case 'review': return <ReviewPage />
      case 'whatif': return <WhatIfPage />
      default: return <OverviewPage />
    }
  }

  return (
    <>
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon">⚙</div>
          <div>
            <h1>MachineGuard</h1>
            <span className="version">Predictive Maintenance v2.0</span>
          </div>
        </div>

        <div className="nav-section">
          <div className="nav-section-title">Monitoring</div>
          {NAV_ITEMS.slice(0, 2).map(item => (
            <button
              key={item.id}
              className={`nav-link ${activePage === item.id ? 'active' : ''}`}
              onClick={() => setActivePage(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </div>

        <div className="nav-section">
          <div className="nav-section-title">Intelligence</div>
          {NAV_ITEMS.slice(2).map(item => (
            <button
              key={item.id}
              className={`nav-link ${activePage === item.id ? 'active' : ''}`}
              onClick={() => setActivePage(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </div>

        <div style={{ marginTop: 'auto', padding: '16px 12px', borderTop: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="live-indicator"></span>
            <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>System Online</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        {renderPage()}
      </main>
    </>
  )
}

export default App
