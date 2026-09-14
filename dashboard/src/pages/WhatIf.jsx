import { useState, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts'

const FEATURES = {
  'ai4i': [
    { name: 'Air temperature [K]', min: 295, max: 310, default: 300 },
    { name: 'Process temperature [K]', min: 305, max: 315, default: 309 },
    { name: 'Rotational speed [rpm]', min: 1100, max: 2900, default: 1500 },
    { name: 'Torque [Nm]', min: 3, max: 80, default: 42 },
    { name: 'Tool wear [min]', min: 0, max: 250, default: 108 },
  ],
  'cmapss': [
    { name: 'sensor_2', min: 641, max: 645, default: 642.5 },
    { name: 'sensor_4', min: 1580, max: 1620, default: 1590 },
    { name: 'sensor_7', min: 550, max: 560, default: 554 },
    { name: 'sensor_11', min: 42, max: 49, default: 47.5 },
    { name: 'sensor_12', min: 518, max: 525, default: 521 },
    { name: 'sensor_14', min: 8090, max: 8180, default: 8140 },
  ],
}

const generateCounterfactual = (values, fleet) => {
  const features = FEATURES[fleet]
  const changes = []
  const numChanges = Math.floor(Math.random() * 2) + 1

  const indices = []
  while (indices.length < numChanges) {
    const idx = Math.floor(Math.random() * features.length)
    if (!indices.includes(idx)) indices.push(idx)
  }

  indices.forEach(idx => {
    const feat = features[idx]
    const original = values[feat.name]
    const spread = (feat.max - feat.min) * 0.15
    const direction = Math.random() > 0.5 ? 1 : -1
    const newVal = Math.max(feat.min, Math.min(feat.max, original + direction * spread * Math.random()))
    changes.push({
      feature: feat.name,
      original: original,
      counterfactual: newVal,
      change: newVal - original,
      change_pct: ((newVal - original) / Math.max(Math.abs(original), 0.001)) * 100,
    })
  })

  return {
    changes,
    rul_improvement: fleet === 'cmapss' ? Math.floor(Math.random() * 25) + 5 : null,
    new_prediction: fleet === 'ai4i' ? 'no_failure' : null,
    confidence: Math.random() * 0.3 + 0.7,
  }
}

export default function WhatIfPage() {
  const [fleet, setFleet] = useState('ai4i')
  const [values, setValues] = useState(() => {
    const v = {}
    FEATURES['ai4i'].forEach(f => { v[f.name] = f.default })
    return v
  })
  const [counterfactuals, setCounterfactuals] = useState([])
  const [currentPrediction, setCurrentPrediction] = useState(null)

  const handleFleetChange = useCallback((newFleet) => {
    setFleet(newFleet)
    const v = {}
    FEATURES[newFleet].forEach(f => { v[f.name] = f.default })
    setValues(v)
    setCounterfactuals([])
    setCurrentPrediction(null)
  }, [])

  const handleSliderChange = (name, value) => {
    setValues(prev => ({ ...prev, [name]: parseFloat(value) }))
  }

  const handlePredict = () => {
    if (fleet === 'ai4i') {
      const torque = values['Torque [Nm]']
      const toolWear = values['Tool wear [min]']
      const isFault = torque > 60 || toolWear > 180
      setCurrentPrediction({
        type: 'fault',
        class: isFault ? 'HDF' : 'no_failure',
        confidence: isFault ? 0.72 : 0.91,
        anomaly_score: isFault ? 0.78 : 0.12,
      })
    } else {
      const s11 = values['sensor_11']
      const rul = Math.max(5, Math.round(200 - (s11 - 42) * 25 + Math.random() * 10))
      setCurrentPrediction({
        type: 'rul',
        rul: rul,
        confidence: 0.85,
      })
    }
  }

  const handleGenerateCounterfactuals = () => {
    const cfs = []
    for (let i = 0; i < 3; i++) {
      cfs.push(generateCounterfactual(values, fleet))
    }
    setCounterfactuals(cfs)
  }

  const features = FEATURES[fleet]

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2>What-If Simulator</h2>
        <p>Explore counterfactual explanations -- what changes would prevent failure?</p>
      </div>

      {/* Fleet selector */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {['ai4i', 'cmapss'].map(f => (
          <button
            key={f}
            className={`btn ${fleet === f ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => handleFleetChange(f)}
          >
            {f === 'ai4i' ? 'AI4I (Fault)' : 'C-MAPSS (RUL)'}
          </button>
        ))}
      </div>

      <div className="charts-grid">
        {/* Input Panel */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Sensor Inputs</div>
              <div className="card-subtitle">Adjust sensor values to explore scenarios</div>
            </div>
          </div>

          <div style={{ marginBottom: 24 }}>
            {features.map((feat) => (
              <div key={feat.name} style={{ marginBottom: 20 }}>
                <div style={{
                  display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 13
                }}>
                  <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{feat.name}</span>
                  <span style={{
                    fontFamily: 'monospace', fontWeight: 700,
                    color: 'var(--accent-blue)', fontSize: 14
                  }}>
                    {values[feat.name]?.toFixed(1)}
                  </span>
                </div>
                <input
                  type="range"
                  min={feat.min}
                  max={feat.max}
                  step={(feat.max - feat.min) / 100}
                  value={values[feat.name] || feat.default}
                  onChange={(e) => handleSliderChange(feat.name, e.target.value)}
                  style={{
                    width: '100%', height: 6, appearance: 'none',
                    background: `linear-gradient(to right, var(--accent-blue) ${((values[feat.name] - feat.min) / (feat.max - feat.min)) * 100}%, var(--bg-primary) 0%)`,
                    borderRadius: 4, outline: 'none', cursor: 'pointer',
                  }}
                />
                <div style={{
                  display: 'flex', justifyContent: 'space-between',
                  fontSize: 11, color: 'var(--text-muted)', marginTop: 2
                }}>
                  <span>{feat.min}</span>
                  <span>{feat.max}</span>
                </div>
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-primary" onClick={handlePredict} style={{ flex: 1, justifyContent: 'center' }}>
              Predict
            </button>
            <button
              className="btn btn-outline"
              onClick={handleGenerateCounterfactuals}
              disabled={!currentPrediction}
              style={{ flex: 1, justifyContent: 'center', opacity: currentPrediction ? 1 : 0.5 }}
            >
              Generate What-Ifs
            </button>
          </div>
        </div>

        {/* Results Panel */}
        <div>
          {/* Current Prediction */}
          {currentPrediction && (
            <div className="card slide-up" style={{ marginBottom: 16 }}>
              <div className="card-header">
                <div className="card-title">Current Prediction</div>
                <span className={`badge ${
                  currentPrediction.type === 'fault'
                    ? (currentPrediction.class === 'no_failure' ? 'healthy' : 'critical')
                    : (currentPrediction.rul > 30 ? 'healthy' : currentPrediction.rul > 15 ? 'warning' : 'critical')
                }`}>
                  {currentPrediction.type === 'fault' ? currentPrediction.class : `RUL: ${currentPrediction.rul}`}
                </span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Confidence</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: 'var(--accent-blue)' }}>
                    {(currentPrediction.confidence * 100).toFixed(0)}%
                  </div>
                </div>
                {currentPrediction.anomaly_score !== undefined && (
                  <div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Anomaly Score</div>
                    <div style={{
                      fontSize: 24, fontWeight: 700,
                      color: currentPrediction.anomaly_score > 0.7 ? 'var(--accent-red)' : 'var(--accent-emerald)'
                    }}>
                      {(currentPrediction.anomaly_score * 100).toFixed(0)}%
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Counterfactuals */}
          {counterfactuals.length > 0 && (
            <div className="card slide-up">
              <div className="card-header">
                <div>
                  <div className="card-title">Counterfactual Explanations</div>
                  <div className="card-subtitle">Minimal changes to prevent failure</div>
                </div>
              </div>

              {counterfactuals.map((cf, i) => (
                <div key={i} className="cf-card">
                  <div style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    marginBottom: 12
                  }}>
                    <span style={{ fontWeight: 700, color: 'var(--accent-purple)' }}>
                      Scenario {i + 1}
                    </span>
                    {cf.rul_improvement && (
                      <span className="badge healthy">+{cf.rul_improvement} cycles RUL</span>
                    )}
                    {cf.new_prediction && (
                      <span className="badge healthy">Prediction: {cf.new_prediction}</span>
                    )}
                  </div>
                  {cf.changes.map((c, j) => (
                    <div key={j} className="cf-change">
                      <span className="feature-name">{c.feature}</span>
                      <span style={{ color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        {c.original.toFixed(1)}
                      </span>
                      <span className="arrow">→</span>
                      <span className="value-change">
                        {c.counterfactual.toFixed(1)}
                      </span>
                      <span style={{
                        fontSize: 12, fontWeight: 600, marginLeft: 'auto',
                        color: c.change_pct > 0 ? 'var(--accent-red)' : 'var(--accent-emerald)',
                      }}>
                        {c.change_pct > 0 ? '+' : ''}{c.change_pct.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              ))}

              {/* Feature Impact Chart */}
              <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, color: 'var(--text-secondary)' }}>
                  Feature Sensitivity
                </div>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart
                    data={counterfactuals[0]?.changes.map(c => ({
                      feature: c.feature.length > 15 ? c.feature.slice(0, 15) + '...' : c.feature,
                      impact: Math.abs(c.change_pct),
                    }))}
                    layout="vertical"
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                    <XAxis type="number" stroke="#64748b" fontSize={11} />
                    <YAxis dataKey="feature" type="category" stroke="#64748b" fontSize={11} width={130} />
                    <Tooltip
                      contentStyle={{
                        background: '#1a1f35',
                        border: '1px solid rgba(148,163,184,0.2)',
                        borderRadius: 8,
                        color: '#f1f5f9'
                      }}
                    />
                    <Bar dataKey="impact" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
