# MachineGuard — Stage 1: Baseline Models & API

Production-grade predictive maintenance system with multi-fleet machine intelligence.

## What Stage 1 Delivers

- **Two ML model families** trained on different failure physics:
  - **RUL Regressor** (C-MAPSS FD001): Predicts remaining useful life of turbofan engines using XGBoost with rolling sensor features. Evaluated with NASA's asymmetric scoring function.
  - **Fault Classifier** (AI4I 2020): Multi-class failure mode classification with SMOTE for class imbalance. Evaluated on per-class recall.
- **SHAP Explainability** for both models via `/explain` API endpoints.
- **Schema-agnostic data pipeline**: Both datasets normalize into a common `MachineReading` schema with JSONB sensor storage — different machine types coexist without schema changes.
- **FastAPI backend** with `/predict/rul`, `/predict/fault`, `/explain/rul`, `/explain/fault` endpoints.
- **TimescaleDB schema** prepared for Stage 2 streaming ingestion.
- **Docker Compose** for local orchestration (TimescaleDB + MLflow + API).

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Download Datasets

```bash
python scripts/download_datasets.py
```

Downloads NASA C-MAPSS (~7MB) and UCI AI4I 2020 (~500KB) into `data/`.

### 3. Train Models

```bash
python scripts/train_models.py
```

Trains both models and saves artifacts to `models/artifacts/`. Takes ~2-5 minutes.

### 4. Run the API

```bash
uvicorn machineguard.api.app:app --reload
```

API docs at http://localhost:8000/docs

### 5. Test

```bash
pytest tests/ -v
```

## Docker Compose (Full Stack)

```bash
docker-compose up -d
```

Services:
- **TimescaleDB**: localhost:5432
- **MLflow**: http://localhost:5001
- **API**: http://localhost:8000

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with model status |
| POST | `/api/v1/predict/rul` | Predict remaining useful life |
| POST | `/api/v1/predict/fault` | Classify failure mode |
| POST | `/api/v1/explain/rul` | SHAP explanation for RUL prediction |
| POST | `/api/v1/explain/fault` | SHAP explanation for fault prediction |

### Example: Predict RUL

```bash
curl -X POST http://localhost:8000/api/v1/predict/rul \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "turbofan_42",
    "sensor_readings": {
      "sensor_2": 642.15, "sensor_3": 1589.70, "sensor_4": 1400.60,
      "sensor_7": 554.36, "sensor_8": 2388.02, "sensor_9": 9046.19,
      "sensor_11": 47.47, "sensor_12": 521.66, "sensor_13": 2388.02,
      "sensor_14": 8138.62, "sensor_15": 8.4195, "sensor_17": 392,
      "sensor_20": 39.06, "sensor_21": 23.42
    }
  }'
```

## Project Structure

```
machineguard/
├── data/          # Data loaders & common schema
├── features/      # Feature engineering (rolling stats, physics features)
├── models/        # Model training, inference & SHAP
├── evaluation/    # Metrics (asymmetric RUL score, per-class recall)
├── db/            # TimescaleDB ORM models
└── api/           # FastAPI endpoints
```

## Scoped Trade-offs

| Decision | Choice | Rationale | Production Alternative |
|----------|--------|-----------|----------------------|
| C-MAPSS subset | FD001 only | Single operating condition, simplest validation | All 4 subsets |
| RUL cap | 125 cycles | Standard practice, avoids flat early-life region | Learned piecewise target |
| Multi-class resolution | Rarest mode wins | Better signal for rare classes | Per-mode binary classifiers |
| SHAP method | TreeExplainer | Exact & fast for tree models | KernelSHAP for model-agnostic |

## Roadmap

- **Stage 2**: Live data ingestion via streaming replay + TimescaleDB
- **Stage 3**: Human-in-the-loop feedback loop + active learning retrain
- **Stage 4**: Fleet fault fingerprinting (autoencoder embeddings + cross-machine alert)
- **Stage 5**: React dashboard + counterfactual "what-if" simulator
