# Real-Time Inference Pipeline with Drift Detection

A streaming ML pipeline that ingests events via Kafka, runs real-time inference, stores predictions in PostgreSQL, and continuously monitors for data drift using statistical tests.

## Architecture

```
Data Producer (synthetic events with drift injection)
        │
        ▼
    Kafka: raw-events
        │
        ▼
    Inference Worker (GradientBoosting model)
        │
        ▼
    Kafka: predictions
        │
        ├──────────────────────┐
        ▼                      ▼
    DB Writer              Drift Detector
        │                      │
        ▼                      ▼
    PostgreSQL          KS Test + PSI Analysis
                               │
                               ▼
                        Kafka: drift-alerts
                               │
                               ▼
                     Prometheus + Grafana Dashboard
```

All services export Prometheus metrics, scraped every 15 seconds and visualized in Grafana.

## What This Demonstrates

| Skill | Where |
|-------|-------|
| Stream processing | Kafka consumers with consumer groups, manual offset commits |
| ML serving | Real-time scikit-learn inference with sub-1ms latency |
| Statistical analysis | Kolmogorov-Smirnov test, Population Stability Index |
| Data engineering | Batched PostgreSQL writes, JSONB storage, idempotent inserts |
| Observability | Prometheus metrics, Grafana dashboards |
| Containerization | Multi-service Docker Compose, health checks, dependency ordering |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- ~4 GB RAM for all containers

### 1. Train the model

```bash
pip install -r requirements.txt
python src/producer/train_model.py
```

### 2. Start everything

```bash
docker compose up --build
```

This starts Kafka, PostgreSQL, Prometheus, Grafana, and all four application services with a single command. The producer begins streaming events at 10/sec. After ~8 minutes (5,000 events), feature distributions shift and the drift detector fires alerts.

### 3. Monitor

- Grafana: http://localhost:3000 (admin / admin)
- Prometheus: http://localhost:9090

## Services

| Service | Port | Description |
|---------|------|-------------|
| producer | 8000 | Generates synthetic user-behavior events with configurable drift injection |
| inference-worker | 8001 | Consumes events, runs GradientBoosting predictions |
| db-writer | 8002 | Batched inserts of predictions into PostgreSQL |
| drift-detector | 8003 | Sliding-window KS test + PSI against training distribution |
| kafka | 9092/9094 | Apache Kafka in KRaft mode (no Zookeeper) |
| postgres | 5433 | Prediction storage |
| prometheus | 9090 | Metrics aggregation |
| grafana | 3000 | Dashboards and alerts |

## Drift Detection

The detector maintains a sliding window of the most recent 1,000 feature vectors and compares them against the training reference distribution using two statistical tests.

**Kolmogorov-Smirnov test** measures the maximum distance between two cumulative distribution functions. It is a non-parametric test that detects any distributional shift. Alert threshold: KS statistic > 0.1.

**Population Stability Index (PSI)** measures how much a distribution has shifted using binned log-likelihood ratios. It is the industry standard metric used in financial model monitoring. Alert threshold: PSI > 0.2 (below 0.1 = no drift, 0.1 to 0.2 = moderate, above 0.2 = significant).

When drift is detected, structured alerts are published to the `drift-alerts` Kafka topic with per-feature KS statistics and PSI scores.

## Key Metrics

| Metric | Description |
|--------|-------------|
| `producer_events_total` | Total events produced |
| `inference_latency_seconds` | Per-event inference latency histogram |
| `dbwriter_rows_inserted_total` | Rows written to PostgreSQL |
| `drift_ks_score{feature}` | KS statistic per feature |
| `drift_psi_score{feature}` | PSI score per feature |
| `drift_alerts_total{feature, test}` | Alert count by feature and test type |

## Configuration

All services are configured via environment variables in `docker-compose.yml`.

| Variable | Default | Description |
|----------|---------|-------------|
| `EVENTS_PER_SECOND` | 10 | Producer throughput |
| `DRIFT_AFTER_N_EVENTS` | 5000 | When drift injection begins |
| `WINDOW_SIZE` | 1000 | Drift detector sliding window size |
| `CHECK_INTERVAL_EVENTS` | 500 | Events between drift checks |
| `KS_THRESHOLD` | 0.1 | KS test alert threshold |
| `PSI_THRESHOLD` | 0.2 | PSI alert threshold |
| `BATCH_SIZE` | 50 | DB writer batch size |
| `BATCH_TIMEOUT_SEC` | 5.0 | DB writer flush timeout |

## Project Structure

```
inference-pipeline-with-drift-detection/
├── docker-compose.yml
├── Dockerfile.producer
├── Dockerfile.inference
├── Dockerfile.dbwriter
├── Dockerfile.drift
├── requirements.txt
├── configs/
│   ├── init.sql                  # PostgreSQL schema
│   ├── prometheus.yml            # Prometheus scrape config
│   └── grafana-datasources.yml   # Auto-provision Prometheus + PostgreSQL
├── models/                       # Generated by train_model.py
│   ├── model.joblib
│   ├── reference_data.npz
│   └── metadata.json
└── src/
    ├── config.py                 # Shared configuration
    ├── producer/
    │   ├── producer.py           # Event generator with drift injection
    │   └── train_model.py        # Offline model training
    ├── inference_worker/
    │   └── worker.py             # Real-time inference consumer
    ├── db_writer/
    │   └── writer.py             # Batched PostgreSQL writer
    └── drift_detector/
        └── detector.py           # KS + PSI drift analysis
```