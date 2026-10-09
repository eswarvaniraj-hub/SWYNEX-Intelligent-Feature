"""
train_model.py
Trains the lightweight text classification model using TF-IDF and Logistic Regression.
Saves the trained pipeline into the 'models/' directory.
"""

import json
import os
import sys
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "classifier_pipeline.joblib")
METADATA_OUTPUT_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
RANDOM_STATE = 42


def train():
    print("=" * 65)
    print(" Smart Text Intelligence - Training Pipeline")
    print("=" * 65)

    # 1. Verify Dataset Existence
    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Dataset not found at '{DATA_PATH}'.")
        print("Please ensure 'data/dataset.csv' exists before training.")
        sys.exit(1)

    print(f"\n[1/5] Loading dataset from: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to read dataset CSV: {e}")
        sys.exit(1)

    # Validate dataset structure
    if "text" not in df.columns or "category" not in df.columns:
        print("[ERROR] Dataset must contain 'text' and 'category' columns.")
        sys.exit(1)

    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df[df["category"] != ""]

    total_samples = len(df)
    class_counts = df["category"].value_counts().to_dict()

    print(f"      Total valid training samples: {total_samples}")
    print("      Class distribution:")
    for cat, count in class_counts.items():
        print(f"        - {cat}: {count} samples")

    if total_samples < 20:
        print("[ERROR] Dataset contains fewer than 20 samples. Insufficient data to train.")
        sys.exit(1)

    # 2. Build Pipeline
    print("\n[2/5] Initializing TF-IDF Vectorizer & Logistic Regression Classifier...")
    # TF-IDF captures both unigrams and bigrams, strips common english stopwords, sublinear tf dampens high frequencies
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        max_features=5000,
        strip_accents="unicode",
    )

    # Multi-class Logistic Regression with fixed random_state for reproducibility
    classifier = LogisticRegression(
        C=2.5,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])

    # 3. Stratified Cross-Validation Evaluation
    print("\n[3/5] Performing 5-fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, df["text"], df["category"], cv=cv, scoring="accuracy")
    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()
    print(f"      5-Fold CV Accuracy: {mean_cv * 100:.2f}% (+/- {std_cv * 100:.2f}%)")

    # 4. Train on Full Dataset
    print("\n[4/5] Fitting final pipeline on complete training dataset...")
    start_time = time.time()
    pipeline.fit(df["text"], df["category"])
    elapsed = time.time() - start_time
    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    print(f"      Pipeline fitted successfully in {elapsed:.3f} seconds.")
    print(f"      Extracted vocabulary size: {vocab_size:,} features.")

    # 5. Persist Model and Metadata
    print("\n[5/5] Saving model artifacts to local 'models/' directory...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"      Model saved: {MODEL_OUTPUT_PATH}")

    metadata = {
        "model_type": "TF-IDF + Logistic Regression",
        "random_state": RANDOM_STATE,
        "total_training_samples": total_samples,
        "class_distribution": class_counts,
        "classes": sorted(list(pipeline.classes_)),
        "cv_accuracy_mean": float(mean_cv),
        "cv_accuracy_std": float(std_cv),
        "vocabulary_size": vocab_size,
        "training_time_seconds": round(elapsed, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"      Metadata saved: {METADATA_OUTPUT_PATH}")

    print("\n" + "=" * 65)
    print(" Training completed successfully!")
    print(" You can now run:")
    print("   - python evaluate.py       (to test the trained model)")
    print("   - streamlit run app.py     (to launch the web interface)")
    print("=" * 65)


if __name__ == "__main__":
    train()
