import { useState } from 'react'

const reviewItems = [
  {
    id: 'RV-001', prediction_id: 'P-4821', machine_id: 'ENG-006',
    fleet: 'C-MAPSS', type: 'RUL', predicted_value: 'RUL: 8 cycles',
    reasons: ['critical_rul', 'high_anomaly_score'], priority: 'critical',
    status: 'pending', created_at: '2 min ago',
    shap_top: [
      { feature: 'sensor_11', value: 0.42, direction: 'up' },
      { feature: 'sensor_4', value: 0.31, direction: 'up' },
      { feature: 'sensor_7', value: -0.18, direction: 'down' },
    ],
  },
  {
    id: 'RV-002', prediction_id: 'P-4819', machine_id: 'M-H202',
    fleet: 'AI4I', type: 'Fault', predicted_value: 'HDF (p=0.68)',
    reasons: ['uncertain_confidence', 'rare_failure_mode'], priority: 'high',
    status: 'pending', created_at: '8 min ago',
    shap_top: [
      { feature: 'Torque [Nm]', value: 0.55, direction: 'up' },
      { feature: 'Process temperature [K]', value: 0.33, direction: 'up' },
      { feature: 'Tool wear [min]', value: 0.21, direction: 'up' },
    ],
  },
  {
    id: 'RV-003', prediction_id: 'P-4815', machine_id: 'ENG-003',
    fleet: 'C-MAPSS', type: 'RUL', predicted_value: 'RUL: 34 cycles',
    reasons: ['model_disagreement'], priority: 'medium',
    status: 'pending', created_at: '23 min ago',
    shap_top: [
      { feature: 'sensor_9', value: 0.28, direction: 'up' },
      { feature: 'sensor_14', value: 0.19, direction: 'down' },
    ],
  },
  {
    id: 'RV-004', prediction_id: 'P-4802', machine_id: 'M-L101',
    fleet: 'AI4I', type: 'Fault', predicted_value: 'OSF (p=0.45)',
    reasons: ['uncertain_confidence'], priority: 'medium',
    status: 'reviewed', verdict: 'false_alarm', created_at: '1 hour ago',
    shap_top: [
      { feature: 'Rotational speed [rpm]', value: 0.38, direction: 'down' },
    ],
  },
  {
    id: 'RV-005', prediction_id: 'P-4790', machine_id: 'ENG-001',
    fleet: 'C-MAPSS', type: 'RUL', predicted_value: 'RUL: 12 cycles',
    reasons: ['critical_rul'], priority: 'critical',
    status: 'reviewed', verdict: 'confirmed_fault', created_at: '2 hours ago',
    shap_top: [
      { feature: 'sensor_11', value: 0.51, direction: 'up' },
      { feature: 'sensor_2', value: 0.29, direction: 'up' },
    ],
  },
]

const reasonLabels = {
  critical_rul: 'Critical RUL',
  high_anomaly_score: 'High Anomaly',
  uncertain_confidence: 'Uncertain',
  rare_failure_mode: 'Rare Mode',
  model_disagreement: 'Disagreement',
}

export default function ReviewPage() {
  const [items, setItems] = useState(reviewItems)
  const [filter, setFilter] = useState('all')
  const [expanded, setExpanded] = useState(null)

  const filtered = filter === 'all' ? items :
    filter === 'pending' ? items.filter(i => i.status === 'pending') :
    items.filter(i => i.status === 'reviewed')

  const handleVerdict = (id, verdict) => {
    setItems(prev => prev.map(item =>
      item.id === id ? { ...item, status: 'reviewed', verdict } : item
    ))
  }

  return (
    <div className="animate-in">
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h2>Review Queue</h2>
            <p>Human-in-the-loop review for uncertain and high-stakes predictions</p>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className="badge pending">{items.filter(i => i.status === 'pending').length} Pending</span>
            <span className="badge info">{items.filter(i => i.status === 'reviewed').length} Reviewed</span>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {['all', 'pending', 'reviewed'].map(f => (
          <button
            key={f}
            className={`btn ${filter === f ? 'btn-primary' : 'btn-outline'}`}
            onClick={() => setFilter(f)}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* Review Items */}
      <div className="card">
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Machine</th>
                <th>Prediction</th>
                <th>Routing Reasons</th>
                <th>Priority</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(item => (
                <>
                  <tr key={item.id}
                    style={{ cursor: 'pointer' }}
                    onClick={() => setExpanded(expanded === item.id ? null : item.id)}
                  >
                    <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>{item.id}</td>
                    <td>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{item.machine_id}</div>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{item.fleet} - {item.type}</div>
                    </td>
                    <td style={{ fontWeight: 600 }}>{item.predicted_value}</td>
                    <td>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {item.reasons.map(r => (
                          <span key={r} className="badge warning" style={{ fontSize: 11, padding: '2px 8px' }}>
                            {reasonLabels[r] || r}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${item.priority === 'critical' ? 'critical' : item.priority === 'high' ? 'warning' : 'info'}`}>
                        {item.priority}
                      </span>
                    </td>
                    <td>
                      {item.status === 'reviewed' ? (
                        <span className={`badge ${item.verdict === 'confirmed_fault' ? 'critical' : 'healthy'}`}>
                          {item.verdict === 'confirmed_fault' ? 'Confirmed' : 'False Alarm'}
                        </span>
                      ) : (
                        <span className="badge pending">Pending</span>
                      )}
                    </td>
                    <td>
                      {item.status === 'pending' && (
                        <div className="review-actions">
                          <button className="btn btn-danger" onClick={(e) => { e.stopPropagation(); handleVerdict(item.id, 'confirmed_fault') }}>
                            Confirm
                          </button>
                          <button className="btn btn-outline" onClick={(e) => { e.stopPropagation(); handleVerdict(item.id, 'false_alarm') }}>
                            Dismiss
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                  {expanded === item.id && (
                    <tr key={item.id + '-detail'}>
                      <td colSpan={7} style={{ background: 'var(--bg-card-hover)', padding: 20 }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
                          <div>
                            <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: 'var(--accent-blue)' }}>
                              SHAP Feature Attribution
                            </h4>
                            {item.shap_top.map((s, i) => (
                              <div key={i} style={{
                                display: 'flex', alignItems: 'center', gap: 12,
                                padding: '8px 0', borderBottom: '1px solid var(--border-subtle)'
                              }}>
                                <span style={{ fontWeight: 600, minWidth: 180 }}>{s.feature}</span>
                                <div style={{
                                  flex: 1, height: 8, background: 'var(--bg-primary)', borderRadius: 4, overflow: 'hidden'
                                }}>
                                  <div style={{
                                    width: `${Math.abs(s.value) * 100}%`,
                                    height: '100%',
                                    background: s.value > 0 ? 'var(--accent-red)' : 'var(--accent-emerald)',
                                    borderRadius: 4,
                                    transition: 'width 0.5s ease',
                                  }} />
                                </div>
                                <span style={{
                                  fontFamily: 'monospace', fontSize: 13, minWidth: 50, textAlign: 'right',
                                  color: s.value > 0 ? 'var(--accent-red)' : 'var(--accent-emerald)',
                                }}>
                                  {s.value > 0 ? '+' : ''}{s.value.toFixed(2)}
                                </span>
                              </div>
                            ))}
                          </div>
                          <div>
                            <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: 'var(--accent-purple)' }}>
                              Details
                            </h4>
                            <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 2 }}>
                              <div><strong>Prediction ID:</strong> {item.prediction_id}</div>
                              <div><strong>Created:</strong> {item.created_at}</div>
                              <div><strong>Fleet:</strong> {item.fleet}</div>
                              <div><strong>Model Type:</strong> {item.type}</div>
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feedback Stats */}
      <div className="charts-grid-equal" style={{ marginTop: 24 }}>
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Feedback Statistics</div>
              <div className="card-subtitle">Retrain trigger status</div>
            </div>
          </div>
          <div style={{ padding: '8px 0' }}>
            {[
              { label: 'Total Feedback', value: '47', color: 'var(--accent-blue)' },
              { label: 'Confirmed Faults', value: '28', color: 'var(--accent-red)' },
              { label: 'False Alarms', value: '15', color: 'var(--accent-amber)' },
              { label: 'Correction Rate', value: '31.9%', color: 'var(--accent-purple)' },
            ].map((s, i) => (
              <div key={i} style={{
                display: 'flex', justifyContent: 'space-between', padding: '12px 0',
                borderBottom: i < 3 ? '1px solid var(--border-subtle)' : 'none'
              }}>
                <span style={{ color: 'var(--text-secondary)' }}>{s.label}</span>
                <span style={{ fontWeight: 700, color: s.color }}>{s.value}</span>
              </div>
            ))}
          </div>
          <button className="btn btn-primary" style={{ marginTop: 12, width: '100%', justifyContent: 'center' }}>
            Trigger Retrain
          </button>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Model Registry</div>
              <div className="card-subtitle">Production model versions</div>
            </div>
          </div>
          <div style={{ padding: '8px 0' }}>
            {[
              { name: 'RUL XGBoost', version: 'v20260914_183045', status: 'production', rmse: '18.2' },
              { name: 'Fault XGBoost', version: 'v20260914_183112', status: 'production', recall: '94%' },
            ].map((m, i) => (
              <div key={i} style={{
                background: 'var(--bg-card-hover)', borderRadius: 'var(--radius-sm)',
                padding: 16, marginBottom: 8
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontWeight: 600 }}>{m.name}</span>
                  <span className="badge healthy">Production</span>
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Version: {m.version}
                </div>
                <div style={{ fontSize: 13, color: 'var(--accent-blue)', marginTop: 4 }}>
                  {m.rmse ? `RMSE: ${m.rmse}` : `Recall: ${m.recall}`}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
