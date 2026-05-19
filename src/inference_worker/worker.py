import json
import logging
import os
import sys
import time

import joblib
import numpy as np
from confluent_kafka import Consumer, Producer, KafkaError
from prometheus_client import Counter, Histogram, start_http_server


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    KAFKA_BOOTSTRAP_SERVERS, TOPIC_RAW_EVENTS, TOPIC_PREDICTIONS,
    FEATURE_NAMES, MODEL_VERSION,
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "inference-workers")
METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("inference-worker")

events_consumed = Counter("inference_events_consumed_total", "Events consumed")
events_processed = Counter("inference_events_processed_total", "Events successfully processed")
inference_errors = Counter("inference_errors_total", "Inference errors")
inference_latency = Histogram(
    "inference_latency_seconds",
    "Per-event inference latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)




def load_model(path):
    logger.info("Loading model from %s ...", path)
    model = joblib.load(path)
    # Warm-up inference
    dummy = np.zeros((1, len(FEATURE_NAMES)))
    model.predict(dummy)
    model.predict_proba(dummy)
    logger.info("Model loaded and warmed up.")
    return model


def extract_features(event):
    features = event["features"]
    return np.array([[features[name] for name in FEATURE_NAMES]])


def run():
    start_http_server(METRICS_PORT)
    logger.info("Prometheus metrics on :%d", METRICS_PORT)

    model = load_model(MODEL_PATH)

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([TOPIC_RAW_EVENTS])

    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "linger.ms": 20,
        "compression.type": "lz4",
    })

    logger.info("Inference worker started. Consuming from '%s'...", TOPIC_RAW_EVENTS)
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Consumer error: %s", msg.error())
                continue

            events_consumed.inc()

            try:
                event = json.loads(msg.value().decode())
                X = extract_features(event)

                start = time.perf_counter()
                prediction = int(model.predict(X)[0])
                probas = model.predict_proba(X)[0]
                confidence = float(probas[prediction])
                latency_sec = time.perf_counter() - start

                inference_latency.observe(latency_sec)

                result = {
                    "event_id": event["event_id"],
                    "timestamp": event["timestamp"],
                    "features": event["features"],
                    "prediction": prediction,
                    "confidence": round(confidence, 4),
                    "model_version": MODEL_VERSION,
                    "latency_ms": round(latency_sec * 1000, 3),
                }

                producer.produce(
                    topic=TOPIC_PREDICTIONS,
                    key=event["event_id"].encode(),
                    value=json.dumps(result).encode(),
                )
                producer.poll(0)

                consumer.commit(asynchronous=False)
                events_processed.inc()
            except Exception:
                inference_errors.inc()
                logger.exception("Error processing event")

    except KeyboardInterrupt:
        logger.info("Shutting down inference worker...")
    finally:
        consumer.close()
        producer.flush(timeout=10)

if __name__ == "__main__":
    run()