import { useState, useEffect } from 'react'
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

// Simulated live RUL data
const generateRULHistory = () => {
  const data = []
  let rul = 120
  for (let i = 0; i < 30; i++) {
    rul = Math.max(5, rul - Math.random() * 6 - 1)
    data.push({
      cycle: i * 10,
      rul: Math.round(rul),
      predicted: Math.round(rul + (Math.random() - 0.5) * 8),
      confidence_upper: Math.round(rul + 12),
      confidence_lower: Math.round(Math.max(0, rul - 12)),
    })
  }
  return data
}

const faultDistribution = [
  { name: 'No Failure', value: 145, color: '#10b981' },
  { name: 'Heat Dissipation', value: 18, color: '#ef4444' },
  { name: 'Power Failure', value: 12, color: '#f59e0b' },
  { name: 'Overstrain', value: 8, color: '#8b5cf6' },
  { name: 'Tool Wear', value: 5, color: '#3b82f6' },
]

const anomalyTimeline = Array.from({ length: 24 }, (_, i) => ({
  hour: `${i}:00`,
  anomaly_score: Math.random() * 0.4 + (i > 14 && i < 20 ? 0.3 : 0),
  threshold: 0.7,
}))

const modelPerformance = [
  { metric: 'RUL RMSE', value: 18.2, target: 20, unit: 'cycles' },
  { metric: 'Fault Recall', value: 0.94, target: 0.90, unit: '' },
  { metric: 'Fault F1', value: 0.91, target: 0.85, unit: '' },
  { metric: 'Asymmetric Score', value: 742, target: 800, unit: '' },
]

export default function OverviewPage() {
  const [rulData] = useState(generateRULHistory)
  const [liveCount, setLiveCount] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setLiveCount(c => c + Math.floor(Math.random() * 3) + 1)
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2>System Overview</h2>
        <p>Real-time predictive maintenance monitoring across all fleets</p>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon blue">🏭</div>
          <div className="stat-content">
            <h3>247</h3>
            <p>Active Machines</p>
            <div className="stat-trend positive">+12 this week</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon emerald">✓</div>
          <div className="stat-content">
            <h3>{(1247 + liveCount).toLocaleString()}</h3>
            <p>Predictions Today</p>
            <div className="stat-trend positive">
              <span className="live-indicator"></span> Live
            </div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon amber">⚠</div>
          <div className="stat-content">
            <h3>14</h3>
            <p>Pending Reviews</p>
            <div className="stat-trend negative">+3 from yesterday</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon red">🔴</div>
          <div className="stat-content">
            <h3>3</h3>
            <p>Critical Alerts</p>
            <div className="stat-trend negative">RUL &lt; 15 cycles</div>
          </div>
        </div>
      </div>

      {/* Charts Row 1 */}
      <div className="charts-grid">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">RUL Prediction Trend</div>
              <div className="card-subtitle">Engine #001 - C-MAPSS Fleet</div>
            </div>
            <span className="badge critical">RUL: {rulData[rulData.length - 1].rul} cycles</span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={rulData}>
              <defs>
                <linearGradient id="rulGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
              <XAxis dataKey="cycle" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={12} />
              <Tooltip
                contentStyle={{
                  background: '#1a1f35',
                  border: '1px solid rgba(148,163,184,0.2)',
                  borderRadius: 8,
                  color: '#f1f5f9'
                }}
              />
              <Area type="monotone" dataKey="confidence_upper" stroke="none" fill="#3b82f6" fillOpacity={0.1} />
              <Area type="monotone" dataKey="confidence_lower" stroke="none" fill="#0a0e1a" fillOpacity={1} />
              <Line type="monotone" dataKey="rul" stroke="#3b82f6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="predicted" stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="5 5" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Fault Distribution</div>
              <div className="card-subtitle">AI4I Fleet - Last 24h</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={faultDistribution}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={3}
                dataKey="value"
              >
                {faultDistribution.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: '#1a1f35',
                  border: '1px solid rgba(148,163,184,0.2)',
                  borderRadius: 8,
                  color: '#f1f5f9'
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: 12, color: '#94a3b8' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="charts-grid-equal">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Anomaly Score Timeline</div>
              <div className="card-subtitle">Fleet-wide anomaly detection</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={anomalyTimeline}>
              <defs>
                <linearGradient id="anomalyGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
              <XAxis dataKey="hour" stroke="#64748b" fontSize={11} interval={3} />
              <YAxis stroke="#64748b" fontSize={11} domain={[0, 1]} />
              <Tooltip
                contentStyle={{
                  background: '#1a1f35',
                  border: '1px solid rgba(148,163,184,0.2)',
                  borderRadius: 8,
                  color: '#f1f5f9'
                }}
              />
              <Area type="monotone" dataKey="anomaly_score" stroke="#ef4444" fill="url(#anomalyGrad)" strokeWidth={2} />
              <Line type="monotone" dataKey="threshold" stroke="#f59e0b" strokeDasharray="6 3" strokeWidth={1.5} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Model Performance</div>
              <div className="card-subtitle">Current production models</div>
            </div>
            <span className="badge healthy">All Passing</span>
          </div>
          <div style={{ padding: '8px 0' }}>
            {modelPerformance.map((m, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '12px 0', borderBottom: i < modelPerformance.length - 1 ? '1px solid var(--border-subtle)' : 'none'
              }}>
                <span style={{ fontSize: 14, color: 'var(--text-secondary)' }}>{m.metric}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{
                    fontSize: 18, fontWeight: 700,
                    color: m.metric.includes('RMSE') || m.metric.includes('Asymmetric')
                      ? (m.value <= m.target ? 'var(--accent-emerald)' : 'var(--accent-red)')
                      : (m.value >= m.target ? 'var(--accent-emerald)' : 'var(--accent-red)')
                  }}>
                    {m.unit === 'cycles' ? m.value : (m.value * (m.value < 10 ? 100 : 1)).toFixed(m.value < 10 ? 0 : 0)}{m.value < 10 && m.unit === '' ? '%' : ''} {m.unit}
                  </span>
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    target: {m.target < 10 ? (m.target * 100).toFixed(0) + '%' : m.target} {m.unit}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
