import { useState } from 'react'

const REVIEW_ITEMS = [
  {
    id: 'RV-2847', predId: 'P-4821', machine: 'ENG-006', fleet: 'C-MAPSS',
    prediction: 'RUL: 8 cycles', confidence: 0.87,
    reasons: ['critical_rul', 'anomaly_0.91'],
    status: 'pending', time: '14:32:08',
    shap: [
      { feature: 'sensor_11', value: 0.42 },
      { feature: 'sensor_4', value: 0.31 },
      { feature: 'sensor_7', value: -0.18 },
    ],
  },
  {
    id: 'RV-2846', predId: 'P-4819', machine: 'M-H202', fleet: 'AI4I',
    prediction: 'HDF (p=0.72)', confidence: 0.72,
    reasons: ['uncertain', 'rare_mode'],
    status: 'pending', time: '14:31:45',
    shap: [
      { feature: 'Torque [Nm]', value: 0.55 },
      { feature: 'Process temp', value: 0.33 },
      { feature: 'Tool wear', value: 0.21 },
    ],
  },
  {
    id: 'RV-2845', predId: 'P-4815', machine: 'ENG-003', fleet: 'C-MAPSS',
    prediction: 'RUL: 34 cycles', confidence: 0.81,
    reasons: ['disagreement'],
    status: 'pending', time: '14:29:15',
    shap: [
      { feature: 'sensor_9', value: 0.28 },
      { feature: 'sensor_14', value: 0.19 },
    ],
  },
  {
    id: 'RV-2843', predId: 'P-4802', machine: 'M-L101', fleet: 'AI4I',
    prediction: 'OSF (p=0.45)', confidence: 0.45,
    reasons: ['uncertain'],
    status: 'reviewed', verdict: 'dismissed', time: '13:42:30',
    shap: [{ feature: 'RPM', value: 0.38 }],
  },
  {
    id: 'RV-2841', predId: 'P-4790', machine: 'ENG-001', fleet: 'C-MAPSS',
    prediction: 'RUL: 12 cycles', confidence: 0.84,
    reasons: ['critical_rul'],
    status: 'reviewed', verdict: 'confirmed', time: '12:55:18',
    shap: [
      { feature: 'sensor_11', value: 0.51 },
      { feature: 'sensor_2', value: 0.29 },
    ],
  },
]

const REASON_MAP = {
  critical_rul: 'Critical RUL',
  'anomaly_0.91': 'High anomaly',
  uncertain: 'Low confidence',
  rare_mode: 'Rare failure mode',
  disagreement: 'Model disagreement',
}

export default function ReviewPage() {
  const [items, setItems] = useState(REVIEW_ITEMS)
  const [filter, setFilter] = useState('all')
  const [expanded, setExpanded] = useState(null)

  const filtered = filter === 'all' ? items
    : items.filter(i => i.status === filter)

  const pending = items.filter(i => i.status === 'pending').length
  const reviewed = items.filter(i => i.status === 'reviewed').length

  const handleVerdict = (id, verdict) => {
    setItems(prev => prev.map(it =>
      it.id === id ? { ...it, status: 'reviewed', verdict } : it
    ))
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <div className="page-title">Review Queue</div>
        <div style={{ display: 'flex', gap: 16, fontSize: 13, color: 'var(--text-2)' }}>
          <span><span className="mono" style={{ color: 'var(--amber-text)' }}>{pending}</span> pending</span>
          <span><span className="mono" style={{ color: 'var(--text-1)' }}>{reviewed}</span> reviewed</span>
        </div>
      </div>
      <div style={{ fontSize: 13, color: 'var(--text-2)', marginBottom: 16 }}>
        Predictions routed for human review based on confidence, anomaly, and disagreement thresholds.
      </div>

      <div className="tab-bar">
        {['all', 'pending', 'reviewed'].map(f => (
          <button key={f} className={`tab-item ${filter === f ? 'active' : ''}`} onClick={() => setFilter(f)}>
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      <table className="review-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Machine</th>
            <th>Prediction</th>
            <th>Conf.</th>
            <th>Routing Reason</th>
            <th>Time</th>
            <th>Verdict</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map(item => (
            <>
              <tr key={item.id} onClick={() => setExpanded(expanded === item.id ? null : item.id)} style={{ cursor: 'pointer' }}>
                <td><span className="mono" style={{ color: 'var(--text-2)' }}>{item.id}</span></td>
                <td>
                  <span className="mono" style={{ color: 'var(--text-0)' }}>{item.machine}</span>
                  <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--text-3)' }}>{item.fleet}</span>
                </td>
                <td><span className="mono">{item.prediction}</span></td>
                <td><span className="mono-sm">{(item.confidence * 100).toFixed(0)}%</span></td>
                <td>
                  <span style={{ fontSize: 12, color: 'var(--text-2)' }}>
                    {item.reasons.map(r => REASON_MAP[r] || r).join(', ')}
                  </span>
                </td>
                <td><span className="mono-sm" style={{ color: 'var(--text-3)' }}>{item.time}</span></td>
                <td>
                  {item.status === 'pending' ? (
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button className="verdict-btn confirm" onClick={(e) => { e.stopPropagation(); handleVerdict(item.id, 'confirmed') }}>
                        Confirm
                      </button>
                      <button className="verdict-btn dismiss" onClick={(e) => { e.stopPropagation(); handleVerdict(item.id, 'dismissed') }}>
                        Dismiss
                      </button>
                    </div>
                  ) : (
                    <span className={`verdict-done ${item.verdict}`}>
                      {item.verdict === 'confirmed' ? 'Confirmed' : 'Dismissed'}
                    </span>
                  )}
                </td>
              </tr>
              {expanded === item.id && (
                <tr key={item.id + '-exp'}>
                  <td colSpan={7} style={{ padding: 0, border: 'none' }}>
                    <div className="detail-panel" style={{ margin: '4px 0 8px' }}>
                      <div style={{ padding: 20 }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32 }}>
                          <div>
                            <div className="detail-section-title">Feature Attribution</div>
                            {item.shap.map((s, i) => (
                              <div className="shap-row" key={i}>
                                <span className="shap-feature">{s.feature}</span>
                                <div className="shap-bar-track">
                                  <div className={`shap-bar-fill ${s.value >= 0 ? 'positive' : 'negative'}`}
                                    style={{ width: `${Math.abs(s.value) * 100}%` }} />
                                </div>
                                <span className={`shap-value ${s.value >= 0 ? 'positive' : 'negative'}`}>
                                  {s.value >= 0 ? '+' : ''}{s.value.toFixed(2)}
                                </span>
                              </div>
                            ))}
                          </div>
                          <div>
                            <div className="detail-section-title">Metadata</div>
                            <div style={{ fontSize: 12.5, lineHeight: 2, color: 'var(--text-2)' }}>
                              <div>Prediction ID: <span className="mono" style={{ color: 'var(--text-1)' }}>{item.predId}</span></div>
                              <div>Fleet: <span style={{ color: 'var(--text-1)' }}>{item.fleet}</span></div>
                              <div>Routing: <span style={{ color: 'var(--text-1)' }}>{item.reasons.map(r => REASON_MAP[r] || r).join(', ')}</span></div>
                            </div>
                          </div>
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

      {/* Retrain stats */}
      <hr className="divider" />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32 }}>
        <div>
          <div className="detail-section-title" style={{ marginBottom: 12 }}>Feedback Statistics</div>
          <div style={{ fontSize: 13, lineHeight: 2.2 }}>
            {[
              { label: 'Total verdicts', value: '47' },
              { label: 'Confirmed faults', value: '28' },
              { label: 'False alarms dismissed', value: '15' },
              { label: 'Correction rate', value: '31.9%', color: 'var(--amber-text)' },
            ].map((s, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', padding: '0 0 2px' }}>
                <span style={{ color: 'var(--text-2)' }}>{s.label}</span>
                <span className="mono" style={{ color: s.color || 'var(--text-0)' }}>{s.value}</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="detail-section-title" style={{ marginBottom: 12 }}>Production Models</div>
          <div style={{ fontSize: 13, lineHeight: 2.2 }}>
            {[
              { label: 'RUL regressor', value: 'v20260914', metric: 'RMSE 18.2' },
              { label: 'Fault classifier', value: 'v20260914', metric: 'Recall 94%' },
            ].map((m, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', padding: '0 0 2px' }}>
                <span style={{ color: 'var(--text-2)' }}>{m.label}</span>
                <span>
                  <span className="mono-sm" style={{ color: 'var(--text-2)' }}>{m.value}</span>
                  <span className="mono-sm" style={{ color: 'var(--green-text)', marginLeft: 12 }}>{m.metric}</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
