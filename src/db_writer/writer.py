import json
import logging
import os
import sys
import time

import psycopg2
import psycopg2.extras
from confluent_kafka import Consumer, KafkaError
from prometheus_client import Counter, Histogram, start_http_server


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import KAFKA_BOOTSTRAP_SERVERS, TOPIC_PREDICTIONS, DATABASE_URL



DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://pipeline:pipeline_pass@localhost:5433/predictions_db",
)

CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "db-writers")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
BATCH_TIMEOUT_SEC = float(os.getenv("BATCH_TIMEOUT_SEC", "5.0"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8002"))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("db-writer")

rows_inserted = Counter("dbwriter_rows_inserted_total", "Rows inserted into PostgreSQL")
insert_errors = Counter("dbwriter_insert_errors_total", "Insert errors")
batch_latency = Histogram(
    "dbwriter_batch_insert_seconds",
    "Batch insert latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

def get_db_connection():
    for attempt in range(10):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.autocommit = False
            logger.info("Connected to PostgreSQL.")
            return conn
        except psycopg2.OperationalError:
            logger.warning("DB not ready, retrying in 2s... (attempt %d)", attempt + 1)
            time.sleep(2)
    raise RuntimeError("Could not connect to PostgreSQL after 10 attempts")


def flush_batch(conn, batch):
    if not batch:
        return

    insert_sql = """
        INSERT INTO predictions (event_id, event_timestamp, features, prediction,
                                  confidence, model_version, latency_ms)
        VALUES %s
        ON CONFLICT (event_id) DO NOTHING
    """
    values = [
        (
            row["event_id"],
            row["timestamp"],
            json.dumps(row["features"]),
            row["prediction"],
            row["confidence"],
            row["model_version"],
            row["latency_ms"],
        )
        for row in batch
    ]

    start = time.perf_counter()
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, insert_sql, values)
        conn.commit()
        elapsed = time.perf_counter() - start
        batch_latency.observe(elapsed)
        rows_inserted.inc(len(values))
        logger.info("Inserted %d rows in %.3fs", len(values), elapsed)
    except Exception:
        conn.rollback()
        insert_errors.inc(len(values))
        logger.exception("Batch insert failed")


def run():
    start_http_server(METRICS_PORT)
    logger.info("Prometheus metrics on :%d", METRICS_PORT)

    conn = get_db_connection()

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": CONSUMER_GROUP,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([TOPIC_PREDICTIONS])

    logger.info("DB writer started. Consuming from '%s'...", TOPIC_PREDICTIONS)

    batch = []
    last_flush = time.time()

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is not None and not msg.error():
                record = json.loads(msg.value().decode())
                batch.append(record)

            elif msg is not None and msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error("Consumer error: %s", msg.error())

            now = time.time()
            if len(batch) >= BATCH_SIZE or (batch and now - last_flush >= BATCH_TIMEOUT_SEC):
                flush_batch(conn, batch)
                consumer.commit(asynchronous=False)
                batch.clear()
                last_flush = now

    except KeyboardInterrupt:
        logger.info("Shutting down DB writer...")
        flush_batch(conn, batch)
    finally:
        consumer.close()
        conn.close()



if __name__ == "__main__":
    run()


