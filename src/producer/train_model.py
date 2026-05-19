import json
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FEATURE_NAMES, MODEL_VERSION

# data is all random generation, when using your own data, rewrite all of this

SAMPLE_SIZE = 10_000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

def generate_training_data(n, seed=42):
    rng = np.random.default_rng(seed)
    session_duration = np.maximum(1, rng.normal(120, 40, n))
    pages_viewed = np.maximum(1, rng.normal(5, 2, n)).astype(int)
    click_rate = np.clip(rng.normal(0.15, 0.05, n), 0, 1)
    scroll_depth = np.clip(rng.normal(0.55, 0.2, n), 0, 1)
    time_of_day = np.clip(rng.normal(14, 4, n), 0, 23).astype(int)
    is_mobile = (rng.random(n) < 0.45).astype(int)
    referral_source = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    X = np.column_stack([
        session_duration, pages_viewed, click_rate, scroll_depth,
        time_of_day, is_mobile, referral_source,
    ])
    convert_score = (
        0.3 * (session_duration / 200)
        + 0.25 * scroll_depth
        + 0.25 * (click_rate / 0.3)
        + 0.1 * (pages_viewed / 10)
        + 0.1 * rng.random(n)
    )
    y = (convert_score > 0.5).astype(int)
    return X, y



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating {SAMPLE_SIZE} training samples...")
    X, y = generate_training_data(SAMPLE_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    print("Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n-- Test Set Performance --")
    print(classification_report(y_test, y_pred, target_names=["no_convert", "convert"]))
    model_path = os.path.join(OUTPUT_DIR, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    ref_path = os.path.join(OUTPUT_DIR, "reference_data.npz")
    np.savez(ref_path, X=X_train)
    print(f"Reference data saved to {ref_path}")
    meta = {
        "model_version": MODEL_VERSION,
        "feature_names": FEATURE_NAMES,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "algorithm": "GradientBoostingClassifier",
    }
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()