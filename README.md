# inference-pipeline-with-drift-detection
# Build Guide: Real-Time Inference Pipeline with Drift Detection

**Timeline:** March 10 – April 3 (3.5 weeks)

**What you're building:** A streaming system that ingests events via Kafka, runs them through an ML model in real time, stores predictions in PostgreSQL, and continuously monitors whether the incoming data is drifting from what the model was trained on — all visualized in a live Grafana dashboard.

---

## Prerequisites & Setup (Do this before March 10)

### Install these tools

- **Docker Desktop** — download from [docker.com](https://www.docker.com/products/docker-desktop/). Make sure Docker Compose v2 is included (it is by default now). Give Docker at least 4 GB RAM in settings.
- **Python 3.11+** — you probably have this already. Confirm with `python3 --version`.
- **A virtual environment** — create one in your project folder:

```bash
mkdir drift-pipeline && cd drift-pipeline
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### Python libraries you'll need

```bash
pip install confluent-kafka numpy scikit-learn scipy joblib psycopg2-binary prometheus-client
```

`confluent-kafka` is the Kafka client. `psycopg2-binary` is PostgreSQL. `prometheus-client` lets your services expose metrics. The rest are for ML and stats.

### Verify Kafka runs on your machine

Create a file called `docker-compose.yml` with just Kafka to test:

```yaml
version: "3.9"
services:
  kafka:
    image: bitnami/kafka:3.7
    ports:
      - "9092:9092"
      - "9094:9094"
    environment:
      - KAFKA_CFG_NODE_ID=0
      - KAFKA_CFG_PROCESS_ROLES=broker,controller
      - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=0@kafka:9093
      - KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093,EXTERNAL://:9094
      - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092,EXTERNAL://localhost:9094
      - KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT,EXTERNAL:PLAINTEXT
      - KAFKA_CFG_INTER_BROKER_LISTENER_NAME=PLAINTEXT
```

Run `docker compose up` and make sure it starts without errors. The key thing here: this is **KRaft mode** (no Zookeeper). Bitnami's image handles all the KRaft config for you through those environment variables. Port 9094 is what your local Python scripts will connect to (`EXTERNAL` listener); port 9092 is for container-to-container communication (`PLAINTEXT` listener).

Once it's running, open another terminal and test:

```bash
docker exec -it <kafka-container-name> kafka-topics.sh --bootstrap-server localhost:9092 --create --topic test-topic --partitions 1
docker exec -it <kafka-container-name> kafka-topics.sh --bootstrap-server localhost:9092 --list
```

If you see `test-topic` in the list, you're good. Tear it down with `docker compose down`.

---

## Project Structure

Set up your folders like this from the start:

```
drift-pipeline/
├── docker-compose.yml
├── Dockerfile.producer
├── Dockerfile.inference
├── Dockerfile.dbwriter
├── Dockerfile.drift
├── requirements.txt
├── configs/
│   ├── init.sql                 # PostgreSQL table schema
│   ├── prometheus.yml           # Prometheus scrape targets
│   ├── grafana-datasources.yml  # auto-provision Prometheus + Postgres into Grafana
│   └── grafana-dashboards.yml   # tell Grafana where to find dashboard JSON
├── dashboards/
│   └── pipeline.json            # Grafana dashboard (build this in Week 3)
├── models/                      # generated files — add to .gitignore
├── scripts/
│   └── setup.sh
├── tests/
└── src/
    ├── config.py                # shared constants (topic names, feature list, etc.)
    ├── producer/
    │   ├── producer.py
    │   └── train_model.py
    ├── inference_worker/
    │   └── worker.py
    ├── db_writer/
    │   └── writer.py
    └── drift_detector/
        └── detector.py
```

Create a `src/config.py` early with shared constants: Kafka bootstrap servers, topic names (`raw-events`, `predictions`, `drift-alerts`), database URL, feature names, model version. Every service imports from here so you're not hardcoding strings everywhere.

---

## Week 1: March 10–16

### Goal: Producer + trained model + inference worker, all talking through Kafka

### Step 1: Build the full Docker Compose (Day 1)

Expand your test `docker-compose.yml` to include all infrastructure. Add these services:

**Kafka** — keep what you had, but add `auto.create.topics.enable=false` and a healthcheck:

```yaml
healthcheck:
  test: kafka-topics.sh --bootstrap-server localhost:9092 --list || exit 1
  interval: 10s
  timeout: 5s
  retries: 10
```

**kafka-init** — a one-shot container that waits for Kafka to be healthy, then creates your three topics (`raw-events`, `predictions`, `drift-alerts`). Use `depends_on` with `condition: service_healthy`. This container runs its commands and exits — that's fine.

**PostgreSQL** — use `postgres:16-alpine`. Set up a user/password/db via environment variables. Mount a `configs/init.sql` file to `/docker-entrypoint-initdb.d/init.sql` — Postgres automatically runs SQL files in that directory on first boot. Your init.sql should create the `predictions` table with columns: `event_id (UUID)`, `event_timestamp (TIMESTAMPTZ)`, `features (JSONB)`, `prediction (INTEGER)`, `confidence (DOUBLE PRECISION)`, `model_version (VARCHAR)`, `latency_ms (DOUBLE PRECISION)`. Add a UNIQUE constraint on event_id and indexes on timestamp and model_version.

**Prometheus** — use `prom/prometheus:v2.51.0`. Mount a `configs/prometheus.yml` that scrapes your four app services on their metrics ports (8000–8003).

**Grafana** — use `grafana/grafana:10.4.0`. Mount a datasources provisioning file that auto-adds Prometheus and PostgreSQL as data sources. Mount a dashboards provisioning file that points to your `dashboards/` folder. This way Grafana is fully configured on first boot — no manual clicking.

Run `docker compose up -d kafka postgres prometheus grafana` and verify everything starts. Check Grafana at localhost:3000 (admin/admin), Prometheus at localhost:9090.

### Step 2: Train the model (Day 1-2)

Write `src/producer/train_model.py`. This script:

1. Generates ~10,000 synthetic samples using the **"normal" distribution** (see below for what features to use)
2. Creates a synthetic binary label (e.g., "will this user convert?") based on a weighted combination of features + some noise
3. Trains a `GradientBoostingClassifier` from scikit-learn (100 estimators, max_depth=4)
4. Prints a classification report so you can verify it learned something
5. Saves three files to `models/`:
   - `model.joblib` — the trained model
   - `reference_data.npz` — the training feature matrix (the drift detector compares against this later)
   - `metadata.json` — feature names, model version, training info

**Feature schema (7 features simulating user behavior):**

| Feature | Normal Distribution | Type |
|---------|-------------------|------|
| session_duration_sec | mean=120, std=40 | continuous |
| pages_viewed | mean=5, std=2 | integer |
| click_rate | mean=0.15, std=0.05 | continuous [0,1] |
| scroll_depth | mean=0.55, std=0.2 | continuous [0,1] |
| time_of_day_hour | mean=14, std=4 | integer [0,23] |
| is_mobile | prob=0.45 | binary |
| referral_source_encoded | choices=[0,1,2], weights=[0.5,0.3,0.2] | categorical |

Use `numpy.random.default_rng(seed=42)` for reproducibility. Clip values to valid ranges (e.g., click_rate between 0 and 1, pages_viewed minimum 1).

Run it: `python src/producer/train_model.py`. You should see accuracy around 75-85% — that's fine, the model quality doesn't matter for this project. What matters is that it produces predictions.

### Step 3: Build the producer (Day 2-3)

Write `src/producer/producer.py`. This is a long-running process that:

1. Starts a Prometheus HTTP server on port 8000 (use `prometheus_client.start_http_server`)
2. Creates a `confluent_kafka.Producer` configured with `linger.ms=50` (micro-batching) and `compression.type=lz4`
3. Loops forever, generating one event per `1/EVENTS_PER_SECOND` seconds
4. Each event is a JSON dict: `{"event_id": <uuid>, "timestamp": <iso8601>, "features": {<the 7 features>}}`
5. Events use the **normal** distribution until `DRIFT_AFTER_N_EVENTS` events have been sent, then switch to the **drifted** distribution

**Drifted distributions (what changes):**

| Feature | Drifted Distribution |
|---------|---------------------|
| session_duration_sec | mean=60, std=30 (users bouncing faster) |
| pages_viewed | mean=2, std=1 |
| click_rate | mean=0.05, std=0.03 (tanked) |
| scroll_depth | mean=0.25, std=0.15 |
| time_of_day_hour | mean=2, std=3 (late-night spike) |
| is_mobile | prob=0.85 (mobile ratio jumps) |
| referral_source_encoded | choices=[0,1,2,3], weights=[0.2,0.1,0.2,0.5] (new source appears) |

The key design choice: define both distributions as dicts at the top of the file, and have `generate_event()` accept a `regime` parameter ("normal" or "drifted"). This makes it clean and easy to explain in interviews.

Add two Prometheus counters: `producer_events_total` and `producer_events_drifted_total`.

**Test it:** With Kafka running, run `python src/producer/producer.py`. In another terminal, consume the topic to verify messages are flowing:

```bash
docker exec -it kafka kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic raw-events --from-beginning --max-messages 5
```

You should see JSON events.

### Step 4: Build the inference worker (Day 3-5)

Write `src/inference_worker/worker.py`. This service:

1. Loads the model from `models/model.joblib` on startup. Run a dummy prediction to warm it up.
2. Creates a Kafka consumer in consumer group `"inference-workers"` subscribed to `raw-events`
3. For each message: deserialize JSON → extract features in the correct order → call `model.predict()` and `model.predict_proba()` → measure latency
4. Publishes the result to the `predictions` topic. The result dict should include: event_id, timestamp, features, prediction, confidence, model_version, latency_ms
5. Commits the offset **after** successful processing (set `enable.auto.commit=False`, call `consumer.commit()` manually). This gives you exactly-once semantics.

**Prometheus metrics to add:** `inference_events_consumed_total`, `inference_events_processed_total`, `inference_errors_total`, and a Histogram for `inference_latency_seconds` with buckets like [0.001, 0.005, 0.01, 0.025, 0.05, 0.1].

**Test it:** Run the producer in one terminal, the inference worker in another. Consume from the `predictions` topic to verify predictions are flowing:

```bash
docker exec -it kafka kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic predictions --from-beginning --max-messages 5
```

You should see events with prediction and confidence fields added. By end of Week 1 you should have: producer → Kafka → inference worker → Kafka, all working.

---

## Week 2: March 17–23

### Goal: DB writer + drift detector, full pipeline running

### Step 5: Build the DB writer (Day 1-2)

Write `src/db_writer/writer.py`. This service:

1. Connects to PostgreSQL with retry logic (loop 10 times with 2-second sleeps — Postgres might not be ready instantly)
2. Creates a Kafka consumer in group `"db-writers"` subscribed to `predictions`
3. Accumulates messages into a batch. Flush the batch when either: the batch reaches 50 rows, OR 5 seconds have passed since the last flush
4. Uses `psycopg2.extras.execute_values()` for efficient bulk inserts. Use `ON CONFLICT (event_id) DO NOTHING` for idempotency
5. Commits Kafka offsets after a successful batch insert

**Prometheus metrics:** `dbwriter_rows_inserted_total`, `dbwriter_insert_errors_total`, `dbwriter_batch_insert_seconds` (Histogram).

**Test it:** Run producer + inference worker + db writer. Then query Postgres:

```bash
docker exec -it postgres psql -U pipeline -d predictions_db -c "SELECT COUNT(*) FROM predictions;"
docker exec -it postgres psql -U pipeline -d predictions_db -c "SELECT * FROM predictions LIMIT 3;"
```

### Step 6: Build the drift detector (Day 2-5)

This is the most interesting service. Write `src/drift_detector/detector.py`:

1. Load the reference data from `models/reference_data.npz` (the training features you saved earlier)
2. Create a Kafka consumer in group `"drift-detectors"` subscribed to `predictions`
3. Maintain a **sliding window** (use `collections.deque(maxlen=1000)`) of recent feature vectors
4. Every 500 events (configurable), run drift checks on the window

**The drift checks (this is the core logic):**

For each of the 7 features, compare the reference column against the window column using:

**Kolmogorov-Smirnov test** — `scipy.stats.ks_2samp(reference_column, window_column)`. This returns a KS statistic (0 to 1, where 0 = identical distributions) and a p-value. Alert if KS stat > 0.1.

**Population Stability Index (PSI)** — you'll implement this yourself. The algorithm:
1. Bin both distributions into N equal-width bins (use 10)
2. Calculate the proportion of values in each bin for both reference and current
3. Add a small epsilon (1e-6) to avoid division by zero
4. PSI = Σ (current_pct - reference_pct) × ln(current_pct / reference_pct)

PSI interpretation: < 0.1 = no drift, 0.1–0.2 = moderate drift, > 0.2 = significant drift. Alert if > 0.2.

When drift is detected, publish a structured alert to the `drift-alerts` topic with: which features drifted, their KS statistics, PSI scores, and the window size.

**Prometheus metrics:** Use `Gauge` (not Counter) for `drift_ks_score` and `drift_psi_score`, labeled by feature name. This way Grafana can show the score over time. Also add `drift_alerts_total` as a Counter labeled by feature and test type.

**Test it:** Run the full pipeline. The producer should run in "normal" mode for the first 5,000 events (~8 minutes at 10/sec). Watch the drift detector logs — it should say "no drift detected" on each check. Then after 5,000 events, the producer switches to drifted distributions. Within a minute or two, the detector should start logging "DRIFT DETECTED" with details about which features shifted.

---

## Week 3: March 24–30

### Goal: Grafana dashboards, alerting, the visual payoff

### Step 7: Build the Grafana dashboard (Day 1-3)

Go to localhost:3000 and build your dashboard. Prometheus should already be connected as a data source (from your provisioning file).

**Panels to create:**

1. **Throughput** — `rate(producer_events_total[1m])` — line chart showing events/sec
2. **Inference latency** — `histogram_quantile(0.95, rate(inference_latency_seconds_bucket[1m]))` — p95 latency over time
3. **DB write rate** — `rate(dbwriter_rows_inserted_total[1m])`
4. **KS scores by feature** — query `drift_ks_score` and set the legend to `{{feature}}`. This gives you one line per feature. Add a threshold line at 0.1.
5. **PSI scores by feature** — same idea with `drift_psi_score`, threshold at 0.2
6. **Drift alerts** — `increase(drift_alerts_total[5m])` — a bar chart or stat panel showing recent alerts

Set the dashboard to auto-refresh every 5 seconds. The money shot is panels 4 and 5: you'll see flat lines during normal operation, then scores spike when drift kicks in.

Once you're happy, export the dashboard JSON and save it to `dashboards/pipeline.json`. This way it auto-loads when someone runs `docker compose up`.

### Step 8: Set up Grafana alerts (Day 3-4)

In Grafana, create alert rules:
- KS score > 0.1 for any feature → fires alert
- PSI score > 0.2 for any feature → fires alert
- Inference error rate > 0 → fires alert

These don't need to send emails or Slack messages — just having them configured and visible in the dashboard is enough to show you understand alerting.

---

## Final Days: March 31 – April 3

### Goal: Docker Compose for everything, README, demo-ready

### Step 9: Dockerize all services (Day 1-2)

Write one Dockerfile per service at the project root (e.g., `Dockerfile.producer`). They all follow the same pattern:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/config.py src/config.py
COPY src/producer/ src/producer/    # change per service
CMD ["python", "src/producer/producer.py"]
```

For the inference worker and drift detector, also `COPY models/ models/` since they need the model and reference data.

Add the four app services to `docker-compose.yml` with proper `depends_on` chains (they should all wait for `kafka-init` to complete, and db-writer should also wait for postgres).

**The goal:** `docker compose up --build` starts the entire system from scratch. One command. This is huge for demos.

### Step 10: Polish (Day 2-3)

**README** — write a good one. Include:
- Architecture diagram (the ASCII one from the project doc works fine, or make a Mermaid diagram)
- "What this demonstrates" table mapping skills to where they appear
- Quick start instructions (3 commands max)
- Table of services with ports
- Explanation of drift detection (KS test, PSI, what the thresholds mean)
- Key metrics table
- Configuration knobs

**Record a demo** — use a screen recorder or `asciinema` for terminal. Show:
1. `docker compose up --build`
2. Grafana dashboard with flat drift scores
3. Wait for drift injection to kick in
4. Drift scores spiking, alerts firing

Save it as a GIF in the repo. This is what makes someone stop scrolling on your GitHub.

---

## Key Concepts to Be Ready to Explain in Interviews

**Why Kafka instead of a simple queue?** Consumer groups give you parallel processing with automatic partition assignment. If you need more throughput, add more inference workers and Kafka rebalances automatically. You also get message durability and replay — if the drift detector crashes, it picks up where it left off.

**Why manual offset commits?** If you auto-commit and the worker crashes after committing but before publishing the prediction, you lose that event. Manual commit after processing gives you at-least-once semantics. Combined with idempotent inserts (ON CONFLICT DO NOTHING), you get effectively-exactly-once.

**Why KS test AND PSI?** They catch different things. KS is a non-parametric test that detects any distributional shift but can be sensitive to sample size. PSI is what the financial industry uses — it's more interpretable and directly measures "how much has this distribution moved." Showing both demonstrates you understand the trade-offs.

**Why batched DB writes?** Individual inserts at 10/sec would be fine, but at higher throughput (100+/sec), the connection overhead per insert kills performance. Batching amortizes that cost. It's also a common pattern you'd see in production.

**What would you do differently in production?** Schema registry for Avro instead of JSON. Model A/B testing. Shadow scoring (run new model alongside old, compare). Automated retraining trigger when drift exceeds threshold. Kubernetes instead of Docker Compose. Dead letter queue for failed events.
