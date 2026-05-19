import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from confluent_kafka import Producer
from prometheus_client import Counter, start_http_server


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")
TOPIC = "raw-events"
EVENTS_PER_SECOND = int(os.getenv("EVENTS_PER_SECOND", "10"))
DRIFT_AFTER_N_EVENTS = int(os.getenv("DRIFT_AFTER_N_EVENTS", "5000"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("producer")

events_produced = Counter("producer_events_total", "Total events produced")
events_drifted = Counter("producer_events_drifted_total", "Events produced after drift injection")



DISTRIBUTIONS = {
    "normal": {
        "session_duration_sec": {"mean": 120, "std": 40},
        "pages_viewed":         {"mean": 5, "std": 2},
        "click_rate":           {"mean": 0.15, "std": 0.05},
        "scroll_depth":         {"mean": 0.55, "std": 0.2},
        "time_of_day_hour":     {"mean": 14, "std": 4},
        "is_mobile":            {"prob": 0.45},
        "referral_source_encoded": {"choices": [0, 1, 2], "weights": [0.5, 0.3, 0.2]},
    },
    "drifted": {
        "session_duration_sec": {"mean": 60, "std": 30},
        "pages_viewed":         {"mean": 2, "std": 1},
        "click_rate":           {"mean": 0.05, "std": 0.03},
        "scroll_depth":         {"mean": 0.25, "std": 0.15},
        "time_of_day_hour":     {"mean": 2, "std": 3},
        "is_mobile":            {"prob": 0.85},
        "referral_source_encoded": {"choices": [0, 1, 2, 3], "weights": [0.2, 0.1, 0.2, 0.5]},
    },
}


def generate_event(rng, regime):
    dist = DISTRIBUTIONS[regime]

    features = {
        "session_duration_sec": max(1.0, rng.normal(dist["session_duration_sec"]["mean"],
                                                     dist["session_duration_sec"]["std"])),
        "pages_viewed": max(1, int(rng.normal(dist["pages_viewed"]["mean"],
                                               dist["pages_viewed"]["std"]))),
        "click_rate": float(np.clip(rng.normal(dist["click_rate"]["mean"],
                                                dist["click_rate"]["std"]), 0, 1)),
        "scroll_depth": float(np.clip(rng.normal(dist["scroll_depth"]["mean"],
                                                  dist["scroll_depth"]["std"]), 0, 1)),
        "time_of_day_hour": int(np.clip(rng.normal(dist["time_of_day_hour"]["mean"],
                                                     dist["time_of_day_hour"]["std"]), 0, 23)),
        "is_mobile": int(rng.random() < dist["is_mobile"]["prob"]),
        "referral_source_encoded": int(rng.choice(
            dist["referral_source_encoded"]["choices"],
            p=dist["referral_source_encoded"]["weights"],
        )),
    }

    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
    }

def delivery_callback(err, msg):
    if err:
        logger.error("Delivery failed: %s", err)
    else:
        events_produced.inc()



def run():
    start_http_server(METRICS_PORT)
    logger.info("Prometheus metrics on :%d", METRICS_PORT)

    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "linger.ms": 50,
        "batch.num.messages": 100,
        "compression.type": "lz4",
    })

    rng = np.random.default_rng(seed=42)
    event_count = 0
    sleep_interval = 1.0 / EVENTS_PER_SECOND

    logger.info(
        "Starting producer - %s @ %d events/sec (drift after %d events)",
        TOPIC, EVENTS_PER_SECOND, DRIFT_AFTER_N_EVENTS,
    )

    try:
        while True:
            regime = "drifted" if event_count >= DRIFT_AFTER_N_EVENTS else "normal"
            event = generate_event(rng, regime)

            producer.produce(
                topic=TOPIC,
                key=event["event_id"].encode(),
                value=json.dumps(event).encode(),
            callback=delivery_callback,
            )
            producer.poll(0)

            if regime == "drifted":
                events_drifted.inc()

            event_count += 1
            if event_count % 500 == 0:
                logger.info("Produced %d events (regime=%s)", event_count, regime)

            time.sleep(sleep_interval)

    except KeyboardInterrupt:
        logger.info("Shutting down producer...")
    finally:
        producer.flush(timeout=10)
        logger.info("Producer flushed. Total events: %d", event_count)


if __name__ == "__main__":
    run()