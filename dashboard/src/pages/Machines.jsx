import { useState, useEffect, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Area, ComposedChart
} from 'recharts'

/* ── Realistic machine data ── */

const generateSensorHistory = (baseline, trend, noise, cycles) => {
  const data = []
  for (let i = 0; i < cycles; i++) {
    const degradation = trend * Math.pow(i / cycles, 1.6)
    const n = (Math.random() - 0.5) * noise
    const value = baseline + degradation + n
    const conf = noise * 0.8 + degradation * 0.15
    data.push({
      cycle: i * 5,
      value: parseFloat(value.toFixed(2)),
      upper: parseFloat((value + conf).toFixed(2)),
      lower: parseFloat((value - conf).toFixed(2)),
    })
  }
  return data
}

const MACHINES = [
  {
    id: 'ENG-006', fleet: 'C-MAPSS', type: 'Turbofan',
    status: 'critical', rul: 8, confidence: 0.87, anomaly: 0.91,
    lastReading: '48s', cycles: 192,
    action: 'Immediate inspection required',
    actionLevel: 'urgent',
    recommendation: 'Engine has exceeded degradation threshold on sensor_11 (exhaust gas temperature). Schedule grounding and borescope inspection within 2 operating cycles. Historical fleet data shows 94% correlation with fan blade erosion at this profile.',
    shap: [
      { feature: 'sensor_11', value: 0.42 },
      { feature: 'sensor_4', value: 0.31 },
      { feature: 'sensor_7', value: -0.18 },
      { feature: 'sensor_15', value: 0.14 },
    ],
    telemetry: generateSensorHistory(47.2, 2.4, 0.3, 40),
    sensorLabel: 'Sensor 11 — EGT',
    sensorUnit: 'K',
    threshold: 48.8,
  },
  {
    id: 'ENG-001', fleet: 'C-MAPSS', type: 'Turbofan',
    status: 'critical', rul: 12, confidence: 0.84, anomaly: 0.87,
    lastReading: '1m 12s', cycles: 186,
    action: 'Schedule maintenance within 5 cycles',
    actionLevel: 'urgent',
    recommendation: 'Accelerating degradation on corrected fan speed (sensor_4) and bleed enthalpy (sensor_7). Pattern matches pre-failure trajectory in fleet cluster C2. Schedule compressor wash and fan balance check.',
    shap: [
      { feature: 'sensor_4', value: 0.38 },
      { feature: 'sensor_11', value: 0.29 },
      { feature: 'sensor_2', value: 0.22 },
    ],
    telemetry: generateSensorHistory(1589, 28, 3.5, 38),
    sensorLabel: 'Sensor 4 — Fan Speed',
    sensorUnit: 'rpm',
    threshold: 1612,
  },
  {
    id: 'M-H202', fleet: 'AI4I', type: 'CNC Mill',
    status: 'critical', rul: null, fault: 'HDF', faultProb: 0.72, anomaly: 0.82,
    lastReading: '3m 04s', cycles: null,
    action: 'Reduce torque, inspect coolant',
    actionLevel: 'urgent',
    recommendation: 'Heat dissipation failure detected with 72% confidence. Process temperature delta exceeds 8.2K above nominal. Reduce torque setting by 12% and verify coolant flow rate. Tool wear at 186 min is approaching replacement threshold.',
    shap: [
      { feature: 'Torque [Nm]', value: 0.55 },
      { feature: 'Process temp [K]', value: 0.33 },
      { feature: 'Tool wear [min]', value: 0.21 },
    ],
    telemetry: generateSensorHistory(309, 4.5, 0.6, 35),
    sensorLabel: 'Process Temperature',
    sensorUnit: 'K',
    threshold: 313,
  },
  {
    id: 'ENG-005', fleet: 'C-MAPSS', type: 'Turbofan',
    status: 'warning', rul: 22, confidence: 0.79, anomaly: 0.62,
    lastReading: '2m 18s', cycles: 168,
    action: 'Schedule inspection within 10 cycles',
    actionLevel: 'caution',
    recommendation: 'Moderate degradation trend on HPC outlet temperature. Rate of change has increased 40% over last 20 cycles. Recommend scheduling borescope at next planned downtime.',
    shap: [
      { feature: 'sensor_9', value: 0.28 },
      { feature: 'sensor_14', value: 0.19 },
      { feature: 'sensor_11', value: 0.15 },
    ],
    telemetry: generateSensorHistory(47.1, 1.6, 0.25, 34),
    sensorLabel: 'Sensor 9 — HPC Outlet',
    sensorUnit: 'K',
    threshold: 48.8,
  },
  {
    id: 'ENG-003', fleet: 'C-MAPSS', type: 'Turbofan',
    status: 'warning', rul: 34, confidence: 0.81, anomaly: 0.54,
    lastReading: '4m 02s', cycles: 156,
    action: 'Monitor — trending',
    actionLevel: 'caution',
    recommendation: 'Slow degradation trend detected. No immediate action required. Continue monitoring sensor_14 (bypass ratio) which shows gradual drift.',
    shap: [
      { feature: 'sensor_14', value: 0.22 },
      { feature: 'sensor_11', value: 0.12 },
    ],
    telemetry: generateSensorHistory(8130, 42, 8, 32),
    sensorLabel: 'Sensor 14 — Bypass Ratio',
    sensorUnit: '',
    threshold: 8170,
  },
  {
    id: 'M-M303', fleet: 'AI4I', type: 'CNC Mill',
    status: 'warning', rul: null, fault: null, faultProb: 0.31, anomaly: 0.48,
    lastReading: '5m 11s', cycles: null,
    action: 'Monitor tool wear',
    actionLevel: 'caution',
    recommendation: 'Tool wear approaching 150 min threshold. No fault predicted yet but anomaly score is elevated. Schedule tool replacement at next shift change.',
    shap: [
      { feature: 'Tool wear [min]', value: 0.29 },
      { feature: 'Torque [Nm]', value: 0.11 },
    ],
    telemetry: generateSensorHistory(108, 38, 5, 30),
    sensorLabel: 'Tool Wear',
    sensorUnit: 'min',
    threshold: 150,
  },
  {
    id: 'ENG-002', fleet: 'C-MAPSS', type: 'Turbofan',
    status: 'operational', rul: 78, confidence: 0.91, anomaly: 0.12,
    lastReading: '1m 34s', cycles: 112,
    action: 'No action required',
    actionLevel: '',
    recommendation: 'All sensors within nominal range. Next scheduled maintenance at cycle 150.',
    shap: [], telemetry: generateSensorHistory(47.0, 0.5, 0.2, 23),
    sensorLabel: 'Sensor 11 — EGT', sensorUnit: 'K', threshold: 48.8,
  },
  {
    id: 'ENG-004', fleet: 'C-MAPSS', type: 'Turbofan',
    status: 'operational', rul: 95, confidence: 0.93, anomaly: 0.08,
    lastReading: '3m 45s', cycles: 95,
    action: 'No action required',
    actionLevel: '',
    recommendation: 'Healthy. All degradation indicators stable.',
    shap: [], telemetry: generateSensorHistory(47.1, 0.3, 0.15, 20),
    sensorLabel: 'Sensor 11 — EGT', sensorUnit: 'K', threshold: 48.8,
  },
  {
    id: 'M-L101', fleet: 'AI4I', type: 'CNC Mill',
    status: 'operational', rul: null, fault: null, faultProb: 0.04, anomaly: 0.09,
    lastReading: '2m 22s', cycles: null,
    action: 'No action required',
    actionLevel: '',
    recommendation: 'Operating within normal parameters.',
    shap: [], telemetry: generateSensorHistory(300, 1.2, 0.4, 25),
    sensorLabel: 'Process Temperature', sensorUnit: 'K', threshold: 313,
  },
]

const EVENTS = [
  { time: '14:32:08', type: 'alert', machine: 'ENG-006', desc: 'RUL dropped below 10 cycles' },
  { time: '14:31:45', type: 'predict', machine: 'M-H202', desc: 'Fault detected: HDF (p=0.72)' },
  { time: '14:30:22', type: 'review', machine: 'ENG-001', desc: 'Confirmed as true fault by operator' },
  { time: '14:29:15', type: 'predict', machine: 'ENG-003', desc: 'RUL: 34 cycles (degrading)' },
  { time: '14:28:01', type: 'system', machine: '---', desc: 'Model v20260914 promoted to production' },
  { time: '14:25:33', type: 'predict', machine: 'ENG-005', desc: 'RUL: 22 cycles (anomaly rising)' },
  { time: '14:22:18', type: 'alert', machine: 'ENG-001', desc: 'Anomaly score exceeded 0.85' },
  { time: '14:20:04', type: 'system', machine: '---', desc: 'Streaming ingestion: 247 machines online' },
]

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

function MachineRow({ machine }) {
  const [expanded, setExpanded] = useState(false)

  const predictionText = machine.rul !== null
    ? `RUL: ${machine.rul} cyc`
    : machine.fault
    ? `${machine.fault} p=${machine.faultProb.toFixed(2)}`
    : `OK p=${(1 - machine.faultProb).toFixed(2)}`

  const predictionClass = machine.status === 'critical' ? 'critical'
    : machine.status === 'warning' ? 'warning' : 'healthy'

  const anomalyLevel = machine.anomaly > 0.7 ? 'high'
    : machine.anomaly > 0.4 ? 'medium' : 'low'

  return (
    <>
      <tr onClick={() => setExpanded(!expanded)}>
        <td>
          <span className="machine-id">
            <span className={`status-dot ${machine.status}`} />
            {machine.id}
          </span>
        </td>
        <td><span className="fleet-tag">{machine.fleet}</span></td>
        <td><span className="fleet-tag">{machine.type}</span></td>
        <td><span className={`prediction-value ${predictionClass}`}>{predictionText}</span></td>
        <td>
          <div className="anomaly-cell">
            <div className="anomaly-bar-track">
              <div
                className={`anomaly-bar-fill ${anomalyLevel}`}
                style={{ width: `${machine.anomaly * 100}%` }}
              />
            </div>
            <span className="anomaly-value">{machine.anomaly.toFixed(2)}</span>
          </div>
        </td>
        <td>
          {machine.confidence != null && (
            <span className="mono-sm">{(machine.confidence * 100).toFixed(0)}%</span>
          )}
        </td>
        <td><span className="mono-sm">{machine.lastReading}</span></td>
        <td>
          <span className={`action-text ${machine.actionLevel}`}>{machine.action}</span>
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={8} style={{ padding: 0, border: 'none' }}>
            <DetailPanel machine={machine} />
          </td>
        </tr>
      )}
    </>
  )
}

function DetailPanel({ machine }) {
  return (
    <div className="detail-panel">
      <div className="detail-split">
        <div className="detail-left">
          <div className="detail-section-title">
            {machine.sensorLabel} [{machine.sensorUnit}]
            {machine.cycles && <span style={{ float: 'right', fontWeight: 400, color: '#6e6e76' }}>{machine.cycles} cycles</span>}
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={machine.telemetry} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1f" />
              <XAxis
                dataKey="cycle"
                stroke="#4a4a52"
                fontSize={10}
                fontFamily="'JetBrains Mono', monospace"
                tickLine={false}
                axisLine={{ stroke: '#25252a' }}
              />
              <YAxis
                stroke="#4a4a52"
                fontSize={10}
                fontFamily="'JetBrains Mono', monospace"
                tickLine={false}
                axisLine={{ stroke: '#25252a' }}
                domain={['dataMin - 1', 'dataMax + 1']}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Area
                type="monotone"
                dataKey="upper"
                stroke="none"
                fill="#6e8fad"
                fillOpacity={0.06}
              />
              <Area
                type="monotone"
                dataKey="lower"
                stroke="none"
                fill="#09090b"
                fillOpacity={1}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#a0a0a8"
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 3, fill: '#ececef', stroke: 'none' }}
              />
              <ReferenceLine
                y={machine.threshold}
                stroke="#ef4444"
                strokeDasharray="6 3"
                strokeWidth={1}
                label={{
                  value: `threshold ${machine.threshold}`,
                  position: 'right',
                  fill: '#ef4444',
                  fontSize: 10,
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="detail-right">
          {machine.shap.length > 0 && (
            <>
              <div className="detail-section-title">Fault Attribution</div>
              {machine.shap.map((s, i) => (
                <div className="shap-row" key={i}>
                  <span className="shap-feature">{s.feature}</span>
                  <div className="shap-bar-track">
                    <div
                      className={`shap-bar-fill ${s.value >= 0 ? 'positive' : 'negative'}`}
                      style={{ width: `${Math.abs(s.value) * 100}%` }}
                    />
                  </div>
                  <span className={`shap-value ${s.value >= 0 ? 'positive' : 'negative'}`}>
                    {s.value >= 0 ? '+' : ''}{s.value.toFixed(2)}
                  </span>
                </div>
              ))}
            </>
          )}

          <div className="recommendation-box" style={{ marginTop: machine.shap.length > 0 ? 16 : 0 }}>
            <div className="rec-label">Recommendation</div>
            <p>{machine.recommendation}</p>
          </div>
        </div>
      </div>
    </div>
  )
}

function MachineGroup({ status, label, machines }) {
  if (machines.length === 0) return null
  return (
    <div className="machine-group">
      <div className="group-header">
        <span className={`status-dot ${status}`} />
        <span className={`group-label ${status}`}>{label}</span>
        <span className="group-count">{machines.length} machine{machines.length !== 1 ? 's' : ''}</span>
      </div>
      <table className="machine-table">
        <thead>
          <tr>
            <th>Machine</th>
            <th>Fleet</th>
            <th>Type</th>
            <th>Prediction</th>
            <th>Anomaly</th>
            <th>Conf.</th>
            <th>Last</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {machines.map(m => <MachineRow key={m.id} machine={m} />)}
        </tbody>
      </table>
    </div>
  )
}

export default function MachinesPage() {
  const [now, setNow] = useState(() => new Date().toLocaleTimeString('en-GB'))
  useEffect(() => {
    const t = setInterval(() => setNow(new Date().toLocaleTimeString('en-GB')), 1000)
    return () => clearInterval(t)
  }, [])

  const groups = useMemo(() => ({
    critical: MACHINES.filter(m => m.status === 'critical'),
    warning: MACHINES.filter(m => m.status === 'warning'),
    operational: MACHINES.filter(m => m.status === 'operational'),
  }), [])

  return (
    <div>
      <div className="fleet-bar">
        <div className="fleet-bar-left">
          <strong>Fleet Status</strong>
          <span className="fleet-sep">|</span>
          <span>{MACHINES.length} total</span>
          <span className="fleet-sep">&middot;</span>
          <span className="fleet-count" style={{ color: 'var(--green-text)' }}>{groups.operational.length} operational</span>
          <span className="fleet-sep">&middot;</span>
          <span className="fleet-count" style={{ color: 'var(--amber-text)' }}>{groups.warning.length} warning</span>
          <span className="fleet-sep">&middot;</span>
          <span className="fleet-count" style={{ color: 'var(--red-text)' }}>{groups.critical.length} critical</span>
        </div>
        <div className="fleet-bar-right">{now}</div>
      </div>

      <MachineGroup status="critical" label="Critical" machines={groups.critical} />
      <MachineGroup status="warning" label="Warning" machines={groups.warning} />
      <MachineGroup status="operational" label="Operational" machines={groups.operational} />

      {/* Event Stream */}
      <div className="event-stream">
        <hr className="divider" />
        <div className="detail-section-title" style={{ marginBottom: 8 }}>Event Stream</div>
        {EVENTS.map((e, i) => (
          <div className="event-row" key={i}>
            <span className="event-time">{e.time}</span>
            <span className={`event-type ${e.type}`}>{e.type.toUpperCase()}</span>
            <span className="event-machine">{e.machine}</span>
            <span className="event-desc">{e.desc}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
