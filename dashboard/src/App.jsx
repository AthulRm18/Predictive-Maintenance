import { useState, useEffect, useCallback } from 'react'
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

/* ── Simulated live alerts ── */

const ALERT_POOL = [
  { severity: 'critical', machine: 'ENG-006', message: 'RUL dropped to 8 cycles. Immediate inspection required.' },
  { severity: 'critical', machine: 'ENG-001', message: 'Anomaly score exceeded 0.87. Degradation accelerating.' },
  { severity: 'critical', machine: 'M-H202', message: 'Heat dissipation failure detected. Confidence: 72%.' },
  { severity: 'warning', machine: 'ENG-005', message: 'RUL below 25 cycles. Schedule maintenance within 10 cycles.' },
  { severity: 'warning', machine: 'ENG-003', message: 'Sensor 14 bypass ratio drifting above nominal range.' },
  { severity: 'warning', machine: 'M-M303', message: 'Tool wear approaching 150 min replacement threshold.' },
  { severity: 'critical', machine: 'ENG-006', message: 'Sensor 11 EGT breached threshold: 48.8 K.' },
  { severity: 'warning', machine: 'ENG-005', message: 'Model disagreement detected. Routed to review queue.' },
  { severity: 'critical', machine: 'M-H202', message: 'Torque reading 71 Nm exceeds safe operating limit.' },
  { severity: 'warning', machine: 'ENG-003', message: 'Anomaly score rising: 0.54. Monitor closely.' },
]

let alertCounter = 0

function AlertToast({ alert, onDismiss }) {
  const [exiting, setExiting] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => {
      setExiting(true)
      setTimeout(onDismiss, 300)
    }, 6000)
    return () => clearTimeout(timer)
  }, [onDismiss])

  const handleClick = () => {
    setExiting(true)
    setTimeout(onDismiss, 300)
  }

  return (
    <div className={`alert-toast ${alert.severity} ${exiting ? 'exiting' : ''}`} onClick={handleClick}>
      <div className={`alert-dot ${alert.severity}`} />
      <div className="alert-body">
        <div className="alert-header">
          <span className="alert-machine">{alert.machine}</span>
          <span className="alert-time">{alert.time}</span>
        </div>
        <div className="alert-message">{alert.message}</div>
        <div className={`alert-severity ${alert.severity}`}>
          {alert.severity === 'critical' ? 'CRITICAL' : 'WARNING'}
        </div>
      </div>
    </div>
  )
}

function App() {
  const [page, setPage] = useState('machines')
  const [alerts, setAlerts] = useState([])

  // Fire first alert after 4s, then every 12-20s
  useEffect(() => {
    const fireAlert = () => {
      const template = ALERT_POOL[Math.floor(Math.random() * ALERT_POOL.length)]
      const now = new Date()
      const time = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      alertCounter++
      setAlerts(prev => [
        { ...template, id: alertCounter, time },
        ...prev.slice(0, 4), // keep max 5
      ])
    }

    const firstTimer = setTimeout(fireAlert, 4000)
    const interval = setInterval(fireAlert, 14000 + Math.random() * 6000)
    return () => { clearTimeout(firstTimer); clearInterval(interval) }
  }, [])

  const dismissAlert = useCallback((id) => {
    setAlerts(prev => prev.filter(a => a.id !== id))
  }, [])

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
      {/* Alert notifications */}
      <div className="alert-container">
        {alerts.map(alert => (
          <AlertToast
            key={alert.id}
            alert={alert}
            onDismiss={() => dismissAlert(alert.id)}
          />
        ))}
      </div>

      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>Ferron</h1>
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
