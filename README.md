# Ferron

**Industrial predictive maintenance. Real-time monitoring. Fleet intelligence.**

Ferron is a production-grade predictive maintenance system that monitors machine fleets, predicts failures before they happen, and tells operators exactly what to do about it.

Built with XGBoost, FastAPI, Redis Streams, and a React dashboard designed with the precision of SpaceX mission control.

> **Live Dashboard**: [ferron.vercel.app](https://ferron.vercel.app)

---

## What it does

**For an operator, Ferron answers five questions at a glance:**

1. Which machines need attention?
2. Why are they unhealthy?
3. How severe is the problem?
4. How much useful life remains?
5. What should I do about it?

---

## System Architecture

```
Datasets (C-MAPSS, AI4I)
    │
    ▼
StreamProducer ──► Redis Streams ──► StreamConsumer
                                         │
                                    Rolling Buffers
                                         │
                          ┌──────────────┼──────────────┐
                          ▼              ▼              ▼
                    RUL Regressor   Fault Classifier   Fleet Embedder
                          │              │              │
                          ▼              ▼              ▼
                    ReviewRouter ◄── Anomaly Score ◄── Cross-Machine Warning
                          │
                    ┌─────┼─────┐
                    ▼           ▼
              Auto-approve   HITL Review Queue
                                │
                          FeedbackStore
                                │
                          ModelGate (promote/reject)
                                │
                          Dashboard (React)
```

---

## Features

### Machine Monitoring
- Severity-grouped fleet table (critical → warning → operational)
- Per-machine RUL prediction, fault classification, anomaly score
- Expandable detail panels with sensor telemetry charts
- Threshold lines, confidence bands, realistic degradation curves
- SHAP-based fault attribution bars
- Actionable maintenance recommendations
- Live alert notifications for critical events

### Streaming Ingestion
- Redis Streams replay simulator for both datasets
- Per-machine rolling buffers with configurable window
- WebSocket push for real-time prediction delivery
- Graceful backpressure handling

### Human-in-the-Loop
- ReviewRouter: routes uncertain, high-anomaly, or disagreement predictions
- FeedbackStore: JSONL-backed verdicts with correction rate tracking
- ModelGate: candidate vs production comparison (handles RMSE and Recall)
- RetrainOrchestrator: closed-loop feedback → retrain → evaluate → promote

### Fleet Intelligence
- PCA-based fleet embedding across heterogeneous sensor spaces
- KMeans clustering for operational regime identification
- Cross-machine drift detection toward degraded clusters
- Silhouette scoring for cluster quality validation

### Counterfactual Simulator
- "What-if" sensor perturbation engine
- Finds minimal changes to prevent predicted failures
- Human-readable recommendations ("Reduce Torque by 8.3%")
- Feature sensitivity visualization

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | XGBoost, SMOTE, SHAP |
| Backend | FastAPI, Pydantic |
| Streaming | Redis Streams, WebSockets |
| Dashboard | React, Vite, Recharts |
| Data | NASA C-MAPSS, UCI AI4I 2020 |
| Testing | Pytest (75 tests) |

---

## Quick Start

### Backend

```bash
# Setup
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Download datasets
python scripts/download_datasets.py

# Train models
python scripts/train_models.py

# Run API
uvicorn machineguard.api.app:app --reload
```

### Dashboard

```bash
cd dashboard
npm install
npm run dev
```

### Full Stack (Docker)

```bash
docker-compose up -d
python scripts/run_streaming_demo.py
```

### Tests

```bash
pytest tests/ -v    # 75/75 passing
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health + model status |
| `POST` | `/api/v1/predict/rul` | Predict remaining useful life |
| `POST` | `/api/v1/predict/fault` | Classify failure mode |
| `POST` | `/api/v1/explain/rul` | SHAP explanation for RUL |
| `POST` | `/api/v1/explain/fault` | SHAP explanation for fault |
| `POST` | `/api/v1/review/route` | Route prediction through HITL |
| `GET` | `/api/v1/review/queue` | Get pending reviews |
| `POST` | `/api/v1/review/submit` | Submit human verdict |
| `GET` | `/api/v1/review/stats` | Feedback statistics |
| `GET` | `/api/v1/models/registry` | Model registry status |

---

## Project Structure

```
ferron/
├── machineguard/
│   ├── data/              # Loaders: C-MAPSS, AI4I, common MachineReading schema
│   ├── models/            # XGBoost RUL regressor + fault classifier
│   ├── evaluation/        # Asymmetric RUL score, per-class metrics, model comparison
│   ├── features/          # Rolling stats, physics-derived features
│   ├── streaming/         # Redis producer, consumer, WebSocket broadcaster
│   ├── hitl/              # ReviewRouter, FeedbackStore, ModelGate, RetrainOrchestrator
│   ├── fleet/             # FleetEmbedder (PCA), FleetClusterer, CrossMachineWarning
│   ├── explainability/    # CounterfactualExplainer (DiCE-style)
│   └── api/               # FastAPI app + route handlers
├── dashboard/             # React + Vite + Recharts
│   └── src/pages/         # Machines, Telemetry, Review, Simulator
├── tests/                 # 75 tests across all subsystems
├── scripts/               # Dataset download, model training, streaming demo
└── docker-compose.yml     # Redis + API orchestration
```

---

## Datasets

| Dataset | Source | What it provides |
|---|---|---|
| **NASA C-MAPSS** | Turbofan run-to-failure | 21-sensor multivariate degradation trajectories |
| **UCI AI4I 2020** | Manufacturing process | 6-feature fault classification with 5 failure modes |

---

## Design Philosophy

The dashboard is designed with the visual discipline of SpaceX, Linear, and Apple:

- Near-black palette with restrained status colors (green/amber/red)
- Monospace typography for all telemetry and machine identifiers
- Sharp 6px geometry, thin borders, no decorative elements
- Information-dense tables over card grids
- Engineering plots with units, thresholds, and confidence bands
- Every element serves a purpose

---

## License

MIT
