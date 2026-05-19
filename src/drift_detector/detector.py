import json
import logging
import os
import sys
import time
from collections import deque

import numpy as np
from confluent_kafka import Consumer, Producer, KafkaError
from prometheus_client import Counter, Gauge, start_http_server
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    KAFKA_BOOTSTRAP_SERVERS, TOPIC_PREDICTIONS, TOPIC_DRIFT_ALERTS,
    FEATURE_NAMES,
)

CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "drift-detectors")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "models/reference_data.npz")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "1000"))
CHECK_INTERVAL_EVENTS = int(os.getenv("CHECK_INTERVAL_EVENTS", "500"))
KS_THRESHOLD = float(os.getenv("KS_THRESHOLD", "0.1"))
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8003"))
PSI_NUM_BINS = 10


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("drift-detector")

events_seen = Counter("drift_events_seen_total", "Events seen by drift detector")
checks_performed = Counter("drift_checks_total", "Drift checks performed")
alerts_fired = Counter("drift_alerts_total", "Drift alerts fired", ["feature", "test"])

ks_score_gauge = Gauge("drift_ks_score", "KS statistic per feature", ["feature"])
psi_score_gauge = Gauge("drift_psi_score", "PSI score per feature", ["feature"])


def compute_psi(reference, current, bins=PSI_NUM_BINS):
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    edges = np.linspace(min_val, max_val, bins + 1)

    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=edges)[0].astype(float)

    ref_pct = (ref_counts + 1e-6) / (ref_counts.sum() + bins * 1e-6)
    cur_pct = (cur_counts + 1e-6) / (cur_counts.sum() + bins * 1e-6)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def load_reference_data(path):
    data = np.load(path)
    X_ref = data["X"]
    logger.info("Loaded reference data: %s", X_ref.shape)
    return X_ref


def run():
    start_http_server(METRICS_PORT)
    logger.info("Prometheus metrics on :%d", METRICS_PORT)

    X_ref = load_reference_data(REFERENCE_DATA_PATH)

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([TOPIC_PREDICTIONS])

    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

    window = deque(maxlen=WINDOW_SIZE)
    events_since_last_check = 0

    logger.info(
        "Drift detector started. Window=%d, check every %d events, KS threshold=%.3f, PSI threshold=%.3f",
        WINDOW_SIZE, CHECK_INTERVAL_EVENTS, KS_THRESHOLD, PSI_THRESHOLD,
    )

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error("Consumer error: %s", msg.error())
                continue

            events_seen.inc()
            record = json.loads(msg.value().decode())

            features = record["features"]
            vec = [features[name] for name in FEATURE_NAMES]
            window.append(vec)
            events_since_last_check += 1

            if events_since_last_check >= CHECK_INTERVAL_EVENTS and len(window) >= WINDOW_SIZE // 2:
                events_since_last_check = 0
                checks_performed.inc()
                X_window = np.array(list(window))

                drift_found = False
                drift_details = {}

                for i, feat_name in enumerate(FEATURE_NAMES):
                    ref_col = X_ref[:, i]
                    win_col = X_window[:, i]

                    # KS test
                    ks_stat, ks_pval = stats.ks_2samp(ref_col, win_col)
                    ks_score_gauge.labels(feature=feat_name).set(ks_stat)

                    # PSI
                    psi_val = compute_psi(ref_col, win_col)
                    psi_score_gauge.labels(feature=feat_name).set(psi_val)

                    feature_drift = {}
                    if ks_stat > KS_THRESHOLD:
                        feature_drift["ks"] = {"statistic": round(ks_stat, 4), "p_value": round(ks_pval, 6)}
                        alerts_fired.labels(feature=feat_name, test="ks").inc()
                        drift_found = True

                    if psi_val > PSI_THRESHOLD:
                        feature_drift["psi"] = round(psi_val, 4)
                        alerts_fired.labels(feature=feat_name, test="psi").inc()
                        drift_found = True

                    if feature_drift:
                        drift_details[feat_name] = feature_drift

                if drift_found:
                    alert = {
                        "alert_type": "data_drift",
                        "timestamp": record["timestamp"],
                        "window_size": len(window),
                        "drifted_features": drift_details,
                    }
                    producer.produce(
                        topic=TOPIC_DRIFT_ALERTS,
                        value=json.dumps(alert).encode(),
                    )
                    producer.poll(0)
                    logger.warning("DRIFT DETECTED: %s", json.dumps(drift_details, indent=2))
                else:
                    logger.info("Drift check passed - no drift detected.")

    except KeyboardInterrupt:
        logger.info("Shutting down drift detector...")
    finally:
        consumer.close()
        producer.flush(timeout=10)


if __name__ == "__main__":
    run()