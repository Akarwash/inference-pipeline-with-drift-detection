import os

# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")

TOPIC_RAW_EVENTS = "raw-events"
TOPIC_PREDICTIONS = "predictions"
TOPIC_DRIFT_ALERTS = "drift-alerts"

# PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://pipeline:pipeline_pass@localhost:5433/predictions_db",
)

# Model
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MODEL_VERSION = "v1.0.0"

# Feature schema
FEATURE_NAMES = [
    "session_duration_sec",
    "pages_viewed",
    "click_rate",
    "scroll_depth",
    "time_of_day_hour",
    "is_mobile",
    "referral_source_encoded",
]

NUM_FEATURES = len(FEATURE_NAMES)