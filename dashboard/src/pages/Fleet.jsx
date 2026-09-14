import { useState } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip, ScatterChart, Scatter, XAxis, YAxis,
  CartesianGrid, ZAxis, Cell
} from 'recharts'

const machines = [
  { id: 'ENG-001', fleet: 'C-MAPSS', rul: 12, status: 'critical', anomaly: 0.87, cluster: 2, temp: 542, vibration: 3.2 },
  { id: 'ENG-002', fleet: 'C-MAPSS', rul: 78, status: 'healthy', anomaly: 0.12, cluster: 0, temp: 518, vibration: 1.1 },
  { id: 'ENG-003', fleet: 'C-MAPSS', rul: 34, status: 'warning', anomaly: 0.54, cluster: 1, temp: 531, vibration: 2.4 },
  { id: 'ENG-004', fleet: 'C-MAPSS', rul: 95, status: 'healthy', anomaly: 0.08, cluster: 0, temp: 512, vibration: 0.9 },
  { id: 'ENG-005', fleet: 'C-MAPSS', rul: 22, status: 'warning', anomaly: 0.62, cluster: 1, temp: 538, vibration: 2.8 },
  { id: 'ENG-006', fleet: 'C-MAPSS', rul: 8, status: 'critical', anomaly: 0.91, cluster: 2, temp: 548, vibration: 3.5 },
  { id: 'M-L101', fleet: 'AI4I', rul: null, status: 'healthy', anomaly: 0.15, cluster: 0, fault: 'None', torque: 42, rpm: 1500 },
  { id: 'M-H202', fleet: 'AI4I', rul: null, status: 'critical', anomaly: 0.82, cluster: 2, fault: 'HDF', torque: 71, rpm: 1380 },
  { id: 'M-M303', fleet: 'AI4I', rul: null, status: 'warning', anomaly: 0.48, cluster: 1, fault: 'None', torque: 55, rpm: 1420 },
]

const clusterData = [
  { x: 2.1, y: 3.5, z: 80, cluster: 0, label: 'Healthy' },
  { x: 2.4, y: 3.2, z: 60, cluster: 0, label: 'Healthy' },
  { x: 1.8, y: 3.8, z: 70, cluster: 0, label: 'Healthy' },
  { x: 2.0, y: 3.6, z: 90, cluster: 0, label: 'Healthy' },
  { x: 5.2, y: 1.5, z: 50, cluster: 1, label: 'Degrading' },
  { x: 5.5, y: 1.2, z: 45, cluster: 1, label: 'Degrading' },
  { x: 5.0, y: 1.8, z: 55, cluster: 1, label: 'Degrading' },
  { x: 7.8, y: -0.5, z: 30, cluster: 2, label: 'Critical' },
  { x: 8.1, y: -0.8, z: 25, cluster: 2, label: 'Critical' },
  { x: 7.5, y: -0.2, z: 35, cluster: 2, label: 'Critical' },
]

const CLUSTER_COLORS = ['#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

const radarMetrics = [
  { metric: 'Temperature', healthy: 0.3, current: 0.7 },
  { metric: 'Vibration', healthy: 0.2, current: 0.8 },
  { metric: 'Pressure', healthy: 0.4, current: 0.5 },
  { metric: 'Torque', healthy: 0.3, current: 0.6 },
  { metric: 'RPM Stability', healthy: 0.9, current: 0.4 },
  { metric: 'Power Draw', healthy: 0.5, current: 0.7 },
]

export default function FleetPage() {
  const [selectedMachine, setSelectedMachine] = useState(null)
  const [fleetFilter, setFleetFilter] = useState('all')

  const filtered = fleetFilter === 'all'
    ? machines
    : machines.filter(m => m.fleet === fleetFilter)

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2>Fleet Intelligence Map</h2>
        <p>Cross-machine fingerprinting, clustering, and early warning detection</p>
      </div>

      {/* Fleet filter */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {['all', 'C-MAPSS', 'AI4I'].map(f => (
          <button
            key={f}
            className={`btn ${fleetFilter === f ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => setFleetFilter(f)}
          >
            {f === 'all' ? 'All Fleets' : f}
          </button>
        ))}
      </div>

      {/* Machine Cards Grid */}
      <div className="fleet-grid" style={{ marginBottom: 32 }}>
        {filtered.map(m => (
          <div
            key={m.id}
            className="machine-card"
            onClick={() => setSelectedMachine(m)}
            style={{
              borderColor: selectedMachine?.id === m.id ? 'var(--accent-blue)' : undefined,
              boxShadow: selectedMachine?.id === m.id ? '0 0 20px var(--accent-blue-glow)' : undefined,
            }}
          >
            <div className="machine-header">
              <h4>{m.id}</h4>
              <span className={`badge ${m.status}`}>
                {m.status.charAt(0).toUpperCase() + m.status.slice(1)}
              </span>
            </div>
            <div className="machine-metrics">
              <div className="machine-metric">
                <label>Fleet</label>
                <span style={{ fontSize: 14, color: 'var(--accent-blue)' }}>{m.fleet}</span>
              </div>
              <div className="machine-metric">
                <label>Anomaly</label>
                <span style={{
                  color: m.anomaly > 0.7 ? 'var(--accent-red)' :
                    m.anomaly > 0.4 ? 'var(--accent-amber)' : 'var(--accent-emerald)'
                }}>
                  {(m.anomaly * 100).toFixed(0)}%
                </span>
              </div>
              {m.rul !== null && (
                <div className="machine-metric">
                  <label>RUL</label>
                  <span style={{
                    color: m.rul < 15 ? 'var(--accent-red)' :
                      m.rul < 40 ? 'var(--accent-amber)' : 'var(--accent-emerald)'
                  }}>
                    {m.rul} cyc
                  </span>
                </div>
              )}
              {m.fault && (
                <div className="machine-metric">
                  <label>Fault</label>
                  <span style={{
                    color: m.fault === 'None' ? 'var(--accent-emerald)' : 'var(--accent-red)',
                    fontSize: 14
                  }}>
                    {m.fault}
                  </span>
                </div>
              )}
              <div className="machine-metric">
                <label>Cluster</label>
                <span style={{ color: CLUSTER_COLORS[m.cluster] }}>
                  C{m.cluster}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Fleet Embedding Scatter + Radar */}
      <div className="charts-grid-equal">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Fleet Embedding Space</div>
              <div className="card-subtitle">PCA projection - machines colored by cluster</div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
              <XAxis dataKey="x" name="PC1" stroke="#64748b" fontSize={12} />
              <YAxis dataKey="y" name="PC2" stroke="#64748b" fontSize={12} />
              <ZAxis dataKey="z" range={[40, 120]} />
              <Tooltip
                contentStyle={{
                  background: '#1a1f35',
                  border: '1px solid rgba(148,163,184,0.2)',
                  borderRadius: 8,
                  color: '#f1f5f9'
                }}
                formatter={(value, name) => [value.toFixed(2), name]}
              />
              <Scatter data={clusterData} shape="circle">
                {clusterData.map((entry, idx) => (
                  <Cell key={idx} fill={CLUSTER_COLORS[entry.cluster]} fillOpacity={0.8} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 24, marginTop: 8 }}>
            {['Healthy', 'Degrading', 'Critical'].map((label, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                <div style={{ width: 10, height: 10, borderRadius: '50%', background: CLUSTER_COLORS[i] }} />
                {label}
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Machine Health Radar</div>
              <div className="card-subtitle">
                {selectedMachine ? selectedMachine.id : 'Select a machine to compare'}
              </div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={radarMetrics}>
              <PolarGrid stroke="rgba(148,163,184,0.15)" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#94a3b8' }} />
              <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} />
              <Radar name="Healthy Baseline" dataKey="healthy" stroke="#10b981" fill="#10b981" fillOpacity={0.15} strokeWidth={2} />
              <Radar name="Current Reading" dataKey="current" stroke="#ef4444" fill="#ef4444" fillOpacity={0.2} strokeWidth={2} />
              <Tooltip
                contentStyle={{
                  background: '#1a1f35',
                  border: '1px solid rgba(148,163,184,0.2)',
                  borderRadius: 8,
                  color: '#f1f5f9'
                }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
