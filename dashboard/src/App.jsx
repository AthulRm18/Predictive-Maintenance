import { useState } from 'react'
import './index.css'
import MachinesPage from './pages/Machines'
import TelemetryPage from './pages/Telemetry'
import ReviewPage from './pages/Review'
import SimulatorPage from './pages/Simulator'

const NAV = [
  { group: 'Monitor', items: [
    { id: 'machines', label: 'Machines' },
    { id: 'telemetry', label: 'Telemetry' },
  ]},
  { group: 'Operate', items: [
    { id: 'review', label: 'Review Queue' },
    { id: 'simulator', label: 'Simulator' },
  ]},
]

function App() {
  const [page, setPage] = useState('machines')

  const renderPage = () => {
    switch (page) {
      case 'machines': return <MachinesPage />
      case 'telemetry': return <TelemetryPage />
      case 'review': return <ReviewPage />
      case 'simulator': return <SimulatorPage />
      default: return <MachinesPage />
    }
  }

  return (
    <>
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>MachineGuard</h1>
          <span>v2.0</span>
        </div>

        {NAV.map(group => (
          <div className="nav-group" key={group.group}>
            <div className="nav-group-label">{group.group}</div>
            {group.items.map(item => (
              <button
                key={item.id}
                className={`nav-item ${page === item.id ? 'active' : ''}`}
                onClick={() => setPage(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
        ))}

        <div className="sidebar-footer">
          <div className="sidebar-status">
            <span className="status-pulse" />
            <span>System operational</span>
          </div>
        </div>
      </aside>

      <main className="main">
        {renderPage()}
      </main>
    </>
  )
}

export default App
