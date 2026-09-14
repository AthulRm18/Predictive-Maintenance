import { useState, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer
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

const FEATURES = {
  ai4i: [
    { name: 'Air temperature', unit: 'K', min: 295, max: 310, default: 300 },
    { name: 'Process temperature', unit: 'K', min: 305, max: 315, default: 309 },
    { name: 'Rotational speed', unit: 'rpm', min: 1100, max: 2900, default: 1500 },
    { name: 'Torque', unit: 'Nm', min: 3, max: 80, default: 42 },
    { name: 'Tool wear', unit: 'min', min: 0, max: 250, default: 108 },
  ],
  cmapss: [
    { name: 'sensor_2', unit: 'R', min: 641, max: 645, default: 642.5 },
    { name: 'sensor_4', unit: 'rpm', min: 1580, max: 1620, default: 1590 },
    { name: 'sensor_7', unit: 'kJ/kg', min: 550, max: 560, default: 554 },
    { name: 'sensor_11', unit: 'K', min: 42, max: 49, default: 47.5 },
    { name: 'sensor_12', unit: 'psi', min: 518, max: 525, default: 521 },
    { name: 'sensor_14', unit: '', min: 8090, max: 8180, default: 8140 },
  ],
}

function generateCounterfactuals(values, fleet) {
  const features = FEATURES[fleet]
  const results = []
  const rng = () => Math.random()

  for (let s = 0; s < 3; s++) {
    const numChanges = Math.floor(rng() * 2) + 1
    const indices = new Set()
    while (indices.size < numChanges) indices.add(Math.floor(rng() * features.length))

    const changes = []
    for (const idx of indices) {
      const feat = features[idx]
      const original = values[feat.name]
      const range = feat.max - feat.min
      const direction = rng() > 0.5 ? 1 : -1
      const newVal = Math.max(feat.min, Math.min(feat.max, original + direction * range * 0.12 * rng()))
      changes.push({
        feature: feat.name,
        unit: feat.unit,
        original,
        newValue: newVal,
        change: newVal - original,
        changePct: ((newVal - original) / Math.max(Math.abs(original), 0.001)) * 100,
      })
    }
    results.push({
      changes,
      improvement: fleet === 'cmapss' ? Math.floor(rng() * 28) + 6 : null,
      newClass: fleet === 'ai4i' ? 'no_failure' : null,
    })
  }
  return results
}

export default function SimulatorPage() {
  const [fleet, setFleet] = useState('ai4i')
  const [values, setValues] = useState(() => {
    const v = {}
    FEATURES.ai4i.forEach(f => { v[f.name] = f.default })
    return v
  })
  const [prediction, setPrediction] = useState(null)
  const [counterfactuals, setCounterfactuals] = useState([])

  const handleFleet = useCallback((f) => {
    setFleet(f)
    const v = {}
    FEATURES[f].forEach(feat => { v[feat.name] = feat.default })
    setValues(v)
    setPrediction(null)
    setCounterfactuals([])
  }, [])

  const predict = () => {
    if (fleet === 'ai4i') {
      const torque = values['Torque']
      const wear = values['Tool wear']
      const isFault = torque > 60 || wear > 180
      setPrediction({
        type: 'fault',
        class: isFault ? 'HDF' : 'no_failure',
        probability: isFault ? 0.74 : 0.06,
        anomaly: isFault ? 0.78 : 0.11,
      })
    } else {
      const s11 = values['sensor_11']
      const rul = Math.max(3, Math.round(200 - (s11 - 42) * 28 + (Math.random() - 0.5) * 6))
      setPrediction({
        type: 'rul',
        rul,
        confidence: 0.86,
        anomaly: s11 > 47 ? 0.65 : 0.15,
      })
    }
    setCounterfactuals([])
  }

  const genCF = () => {
    if (!prediction) return
    setCounterfactuals(generateCounterfactuals(values, fleet))
  }

  const features = FEATURES[fleet]

  return (
    <div>
      <div className="page-title">Counterfactual Simulator</div>
      <div style={{ fontSize: 13, color: 'var(--text-2)', marginTop: 2, marginBottom: 16 }}>
        Explore what sensor changes would prevent the predicted failure or extend remaining life.
      </div>

      <div className="tab-bar">
        <button className={`tab-item ${fleet === 'ai4i' ? 'active' : ''}`} onClick={() => handleFleet('ai4i')}>
          AI4I — Fault Classification
        </button>
        <button className={`tab-item ${fleet === 'cmapss' ? 'active' : ''}`} onClick={() => handleFleet('cmapss')}>
          C-MAPSS — RUL Regression
        </button>
      </div>

      <div className="sim-layout">
        {/* Input panel */}
        <div className="sim-panel">
          <div className="detail-section-title" style={{ marginBottom: 16 }}>Sensor Inputs</div>
          {features.map(feat => (
            <div className="sim-input-row" key={feat.name}>
              <div className="sim-input-label">
                <span>{feat.name} <span style={{ color: 'var(--text-3)' }}>[{feat.unit}]</span></span>
                <span>{values[feat.name]?.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min={feat.min}
                max={feat.max}
                step={(feat.max - feat.min) / 200}
                value={values[feat.name] || feat.default}
                onChange={(e) => setValues(prev => ({ ...prev, [feat.name]: parseFloat(e.target.value) }))}
              />
              <div className="sim-range-labels">
                <span>{feat.min}</span>
                <span>{feat.max}</span>
              </div>
            </div>
          ))}
          <div className="sim-actions">
            <button className="btn btn-primary" onClick={predict}>Predict</button>
            <button className="btn" onClick={genCF} disabled={!prediction}>Generate counterfactuals</button>
          </div>
        </div>

        {/* Results panel */}
        <div className="sim-panel">
          {prediction && (
            <>
              <div className="detail-section-title" style={{ marginBottom: 12 }}>Prediction</div>
              <div className="sim-result">
                {prediction.type === 'fault' ? (
                  <>
                    <div className="sim-result-row">
                      <span className="sim-result-label">Predicted class</span>
                      <span className="sim-result-value" style={{
                        color: prediction.class === 'no_failure' ? 'var(--green-text)' : 'var(--red-text)'
                      }}>
                        {prediction.class}
                      </span>
                    </div>
                    <div className="sim-result-row">
                      <span className="sim-result-label">Probability</span>
                      <span className="sim-result-value">{(prediction.probability * 100).toFixed(0)}%</span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="sim-result-row">
                      <span className="sim-result-label">Remaining useful life</span>
                      <span className="sim-result-value" style={{
                        color: prediction.rul < 15 ? 'var(--red-text)' : prediction.rul < 40 ? 'var(--amber-text)' : 'var(--text-0)'
                      }}>
                        {prediction.rul} cycles
                      </span>
                    </div>
                    <div className="sim-result-row">
                      <span className="sim-result-label">Confidence</span>
                      <span className="sim-result-value">{(prediction.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </>
                )}
                <div className="sim-result-row">
                  <span className="sim-result-label">Anomaly score</span>
                  <span className="sim-result-value" style={{
                    color: prediction.anomaly > 0.7 ? 'var(--red-text)' : prediction.anomaly > 0.4 ? 'var(--amber-text)' : 'var(--text-2)'
                  }}>
                    {prediction.anomaly.toFixed(2)}
                  </span>
                </div>
              </div>
            </>
          )}

          {counterfactuals.length > 0 && (
            <>
              <div className="detail-section-title" style={{ marginBottom: 12 }}>Counterfactuals</div>
              {counterfactuals.map((cf, i) => (
                <div className="cf-scenario" key={i}>
                  <div className="cf-header">
                    <span className="cf-label">Scenario {i + 1}</span>
                    {cf.improvement && <span className="cf-outcome">+{cf.improvement} cycles</span>}
                    {cf.newClass && <span className="cf-outcome">{cf.newClass}</span>}
                  </div>
                  {cf.changes.map((c, j) => (
                    <div className="cf-change-row" key={j}>
                      <span className="cf-feat">{c.feature}</span>
                      <span className="mono-sm" style={{ color: 'var(--text-3)' }}>{c.original.toFixed(1)}</span>
                      <span className="cf-arrow">&rarr;</span>
                      <span className="cf-new">{c.newValue.toFixed(1)}</span>
                      <span className={`cf-pct ${c.changePct < 0 ? 'down' : 'up'}`}>
                        {c.changePct > 0 ? '+' : ''}{c.changePct.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              ))}

              <div style={{ marginTop: 16 }}>
                <div className="detail-section-title" style={{ marginBottom: 8 }}>Feature Sensitivity</div>
                <ResponsiveContainer width="100%" height={140}>
                  <BarChart
                    data={counterfactuals[0].changes.map(c => ({
                      feature: c.feature.length > 12 ? c.feature.slice(0, 12) + '..' : c.feature,
                      impact: Math.abs(c.changePct),
                    }))}
                    layout="vertical"
                    margin={{ top: 0, right: 8, bottom: 0, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1f" />
                    <XAxis type="number" stroke="#4a4a52" fontSize={10} fontFamily="'JetBrains Mono', monospace" />
                    <YAxis dataKey="feature" type="category" stroke="#4a4a52" fontSize={10}
                      fontFamily="'JetBrains Mono', monospace" width={100} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="impact" fill="#6e8fad" radius={[0, 2, 2, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {!prediction && !counterfactuals.length && (
            <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-3)', fontSize: 13 }}>
              Adjust sensor values and click Predict to start.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
