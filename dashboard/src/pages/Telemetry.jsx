import { useState } from 'react'
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'

const tooltipStyle = {
  background: '#0f0f12',
  border: '1px solid #25252a',
  borderRadius: 4,
  padding: '8px 12px',
  fontSize: 12,
  fontFamily: "'JetBrains Mono', monospace",
  color: '#ececef',
  boxShadow: 'none',
}

const SENSOR_PROFILES = {
  'ENG-006': {
    label: 'Engine ENG-006',
    fleet: 'C-MAPSS',
    status: 'critical',
    rul: 8,
    sensors: [
      {
        name: 'Sensor 11 — Exhaust Gas Temperature',
        unit: 'K',
        threshold: 48.8,
        data: Array.from({ length: 50 }, (_, i) => {
          const base = 47.2 + 2.0 * Math.pow(i / 50, 1.5)
          const n = (Math.random() - 0.5) * 0.25
          const v = base + n
          return { cycle: i * 4, value: +v.toFixed(2), upper: +(v + 0.4 + i * 0.01).toFixed(2), lower: +(v - 0.4 - i * 0.01).toFixed(2) }
        }),
      },
      {
        name: 'Sensor 4 — Corrected Fan Speed',
        unit: 'rpm',
        threshold: 1612,
        data: Array.from({ length: 50 }, (_, i) => {
          const base = 1589 + 24 * Math.pow(i / 50, 1.4)
          const n = (Math.random() - 0.5) * 3
          const v = base + n
          return { cycle: i * 4, value: +v.toFixed(1), upper: +(v + 4).toFixed(1), lower: +(v - 4).toFixed(1) }
        }),
      },
      {
        name: 'Sensor 7 — Bleed Enthalpy',
        unit: 'kJ/kg',
        threshold: 558,
        data: Array.from({ length: 50 }, (_, i) => {
          const base = 553.5 + 5.5 * Math.pow(i / 50, 1.3)
          const n = (Math.random() - 0.5) * 1.2
          const v = base + n
          return { cycle: i * 4, value: +v.toFixed(1), upper: +(v + 2).toFixed(1), lower: +(v - 2).toFixed(1) }
        }),
      },
      {
        name: 'Sensor 2 — Total Temp at LPC Outlet',
        unit: 'R',
        threshold: 644.5,
        data: Array.from({ length: 50 }, (_, i) => {
          const base = 642.2 + 2.8 * Math.pow(i / 50, 1.2)
          const n = (Math.random() - 0.5) * 0.3
          const v = base + n
          return { cycle: i * 4, value: +v.toFixed(2), upper: +(v + 0.5).toFixed(2), lower: +(v - 0.5).toFixed(2) }
        }),
      },
    ],
    metrics: [
      { label: 'Total Cycles', value: '192', unit: '' },
      { label: 'Predicted RUL', value: '8', unit: 'cycles' },
      { label: 'Anomaly Score', value: '0.91', unit: '' },
      { label: 'Model Confidence', value: '87', unit: '%' },
    ],
    history: [
      { time: '14:32:08', event: 'RUL prediction: 8 cycles', type: 'prediction' },
      { time: '14:20:15', event: 'Anomaly score exceeded 0.85', type: 'alert' },
      { time: '13:45:02', event: 'RUL prediction: 11 cycles', type: 'prediction' },
      { time: '12:30:44', event: 'Entered critical state', type: 'alert' },
      { time: '10:15:22', event: 'RUL prediction: 18 cycles', type: 'prediction' },
      { time: '08:00:00', event: 'Shift start — nominal check', type: 'system' },
    ],
  },
  'M-H202': {
    label: 'Machine M-H202',
    fleet: 'AI4I',
    status: 'critical',
    fault: 'HDF',
    sensors: [
      {
        name: 'Process Temperature',
        unit: 'K',
        threshold: 313,
        data: Array.from({ length: 40 }, (_, i) => {
          const base = 309 + 4.8 * Math.pow(i / 40, 1.6)
          const n = (Math.random() - 0.5) * 0.5
          const v = base + n
          return { cycle: i * 3, value: +v.toFixed(1), upper: +(v + 0.8).toFixed(1), lower: +(v - 0.8).toFixed(1) }
        }),
      },
      {
        name: 'Torque',
        unit: 'Nm',
        threshold: 65,
        data: Array.from({ length: 40 }, (_, i) => {
          const base = 42 + 28 * Math.pow(i / 40, 1.8)
          const n = (Math.random() - 0.5) * 3
          const v = base + n
          return { cycle: i * 3, value: +v.toFixed(1), upper: +(v + 4).toFixed(1), lower: +(v - 4).toFixed(1) }
        }),
      },
      {
        name: 'Tool Wear',
        unit: 'min',
        threshold: 200,
        data: Array.from({ length: 40 }, (_, i) => {
          const v = 5 + i * 4.8
          return { cycle: i * 3, value: +v.toFixed(0), upper: +(v + 5).toFixed(0), lower: +Math.max(0, v - 5).toFixed(0) }
        }),
      },
    ],
    metrics: [
      { label: 'Product Type', value: 'H', unit: '' },
      { label: 'Fault Class', value: 'HDF', unit: '' },
      { label: 'Fault Prob.', value: '0.72', unit: '' },
      { label: 'Anomaly Score', value: '0.82', unit: '' },
    ],
    history: [
      { time: '14:31:45', event: 'Fault detected: HDF (p=0.72)', type: 'alert' },
      { time: '14:15:30', event: 'Anomaly score exceeded 0.70', type: 'alert' },
      { time: '13:50:12', event: 'Torque spike detected', type: 'prediction' },
      { time: '12:00:00', event: 'Tool change — reset wear counter', type: 'system' },
    ],
  },
}

const machineOptions = Object.keys(SENSOR_PROFILES)

function SensorChart({ sensor }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <span className="detail-section-title" style={{ marginBottom: 0 }}>{sensor.name}</span>
        <span className="mono-sm" style={{ color: 'var(--text-3)' }}>[{sensor.unit}]</span>
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={sensor.data} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1f" />
          <XAxis dataKey="cycle" stroke="#4a4a52" fontSize={10} fontFamily="'JetBrains Mono', monospace" tickLine={false} axisLine={{ stroke: '#25252a' }} />
          <YAxis stroke="#4a4a52" fontSize={10} fontFamily="'JetBrains Mono', monospace" tickLine={false} axisLine={{ stroke: '#25252a' }} domain={['dataMin - 1', 'dataMax + 1']} />
          <Tooltip contentStyle={tooltipStyle} />
          <Area type="monotone" dataKey="upper" stroke="none" fill="#6e8fad" fillOpacity={0.06} />
          <Area type="monotone" dataKey="lower" stroke="none" fill="#09090b" fillOpacity={1} />
          <Line type="monotone" dataKey="value" stroke="#a0a0a8" strokeWidth={1.5} dot={false} activeDot={{ r: 3, fill: '#ececef', stroke: 'none' }} />
          <ReferenceLine y={sensor.threshold} stroke="#ef4444" strokeDasharray="6 3" strokeWidth={1}
            label={{ value: `${sensor.threshold}`, position: 'right', fill: '#ef4444', fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function TelemetryPage() {
  const [selected, setSelected] = useState(machineOptions[0])
  const profile = SENSOR_PROFILES[selected]

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <div className="page-title">Telemetry</div>
          <div style={{ fontSize: 13, color: 'var(--text-2)', marginTop: 2 }}>Sensor data and maintenance history</div>
        </div>
      </div>

      {/* Machine selector tabs */}
      <div className="tab-bar">
        {machineOptions.map(id => (
          <button key={id} className={`tab-item ${selected === id ? 'active' : ''}`} onClick={() => setSelected(id)}>
            <span className="mono" style={{ fontSize: 12.5 }}>{id}</span>
            <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--text-3)' }}>{SENSOR_PROFILES[id].fleet}</span>
          </button>
        ))}
      </div>

      {/* Metrics row */}
      <div className="metrics-row">
        {profile.metrics.map((m, i) => (
          <div className="metric-item" key={i}>
            <div className="metric-label">{m.label}</div>
            <div>
              <span className="metric-value" style={{
                color: m.label.includes('Fault') && m.value !== '' ? 'var(--red-text)' :
                  m.label.includes('Anomaly') && parseFloat(m.value) > 0.7 ? 'var(--red-text)' : undefined
              }}>
                {m.value}
              </span>
              {m.unit && <span className="metric-unit">{m.unit}</span>}
            </div>
          </div>
        ))}
      </div>

      {/* Sensor charts */}
      <div style={{ marginTop: 24 }}>
        {profile.sensors.map((sensor, i) => (
          <SensorChart key={i} sensor={sensor} />
        ))}
      </div>

      {/* Maintenance history */}
      <hr className="divider" />
      <div className="detail-section-title" style={{ marginBottom: 8 }}>Maintenance History</div>
      {profile.history.map((h, i) => (
        <div className="event-row" key={i}>
          <span className="event-time">{h.time}</span>
          <span className={`event-type ${h.type}`}>{h.type.toUpperCase()}</span>
          <span className="event-desc">{h.event}</span>
        </div>
      ))}
    </div>
  )
}
