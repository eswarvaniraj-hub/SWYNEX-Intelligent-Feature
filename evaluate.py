"""
evaluate.py
Evaluates the trained text classifier on an independent, labeled test dataset.
Computes accuracy, sample-by-sample results, category classification reports, and a confusion matrix.
"""

import os
import sys
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "classifier_pipeline.joblib")
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "test_dataset.csv")

# Embedded fallback test set (24 samples: 6 per category) in case CSV is unreadable
FALLBACK_TEST_SAMPLES = [
    # Technology (6)
    {"text": "Developers configured automated alerting systems on cloud microservices using Prometheus.", "category": "Technology"},
    {"text": "The cybersecurity team patched critical zero-day vulnerabilities in the corporate network firewall.", "category": "Technology"},
    {"text": "Machine learning engineers fine-tuned transformer weights on domain-specific code datasets.", "category": "Technology"},
    {"text": "The relational database failed over to a secondary replica during peak query traffic.", "category": "Technology"},
    {"text": "Quantum cryptography protocols generate tamper-evident quantum encryption keys.", "category": "Technology"},
    {"text": "Software architects refactored monolithic legacy code into independent RESTful microservices.", "category": "Technology"},
    # Education (6)
    {"text": "University students submitted their term papers before the final semester examination deadline.", "category": "Education"},
    {"text": "The school principal introduced a blended learning initiative with interactive digital textbooks.", "category": "Education"},
    {"text": "Doctoral candidates presented their research findings during the international academic symposium.", "category": "Education"},
    {"text": "The mathematics department organized tutoring workshops for students preparing for calculus exams.", "category": "Education"},
    {"text": "Kindergarten teachers focus on early literacy, phonics, and socialization activities.", "category": "Education"},
    {"text": "Academic advisors assisted undergraduate students in declaring majors and choosing course electives.", "category": "Education"},
    # Sports (6)
    {"text": "The basketball center pulled down fifteen rebounds and scored twenty points in the playoff victory.", "category": "Sports"},
    {"text": "The tennis player served an ace on match point to claim the Grand Slam trophy.", "category": "Sports"},
    {"text": "The football referee reviewed video footage before awarding a penalty kick in extra time.", "category": "Sports"},
    {"text": "Olympic sprinters completed rigorous interval sprint training to improve starting block acceleration.", "category": "Sports"},
    {"text": "The hockey team scored two power play goals in the final period to overcome a two-goal deficit.", "category": "Sports"},
    {"text": "Golfers adjusted their swing mechanics to compensate for crosswinds on the championship links.", "category": "Sports"},
    # Health (6)
    {"text": "Patients suffering from hypertension were advised to reduce sodium intake and monitor arterial blood pressure.", "category": "Health"},
    {"text": "The clinical study evaluated the efficacy of a novel antiviral medication in reducing hospital recovery time.", "category": "Health"},
    {"text": "Dermatologists recommend daily application of broad-spectrum sunscreen to guard against malignant melanoma.", "category": "Health"},
    {"text": "Physical therapy sessions helped restore shoulder mobility and reduce inflammation following rotator cuff surgery.", "category": "Health"},
    {"text": "Cognitive behavioral therapy provides actionable coping strategies to alleviate severe generalized anxiety.", "category": "Health"},
    {"text": "Registered dietitians emphasize balanced macronutrient ratios to manage type 2 diabetes and regulate glucose.", "category": "Health"},
]


def load_test_data() -> pd.DataFrame:
    """Loads the test dataset from CSV or falls back to embedded data."""
    if os.path.exists(TEST_DATA_PATH):
        try:
            df = pd.read_csv(TEST_DATA_PATH)
            if "text" in df.columns and "category" in df.columns and len(df) >= 20:
                return df
        except Exception as e:
            print(f"[WARNING] Could not load test CSV: {e}. Using embedded test dataset.")

    return pd.DataFrame(FALLBACK_TEST_SAMPLES)


def evaluate():
    print("=" * 80)
    print(" Smart Text Intelligence - Model Evaluation")
    print("=" * 80)

    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Trained model file not found at: {MODEL_PATH}")
        print("Please train the model first by running:")
        print("    python train_model.py\n")
        sys.exit(1)

    print(f"\n[1/4] Loading model from: {MODEL_PATH}")
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("The model file might be corrupted. Regenerate it with 'python train_model.py'.")
        sys.exit(1)

    # 2. Load test samples
    test_df = load_test_data()
    total_tests = len(test_df)
    print(f"[2/4] Loaded {total_tests} test samples across all 4 categories.")
    print("-" * 80)

    # 3. Run predictions
    print("\n[3/4] Individual Sample Evaluation Results:")
    print("-" * 80)
    print(f"{'#':<3} | {'Expected':<12} | {'Predicted':<12} | {'Result':<6} | {'Sample Text':<42}")
    print("-" * 80)

    y_true = []
    y_pred = []
    correct_count = 0
    incorrect_count = 0

    for idx, row in test_df.iterrows():
        text = str(row["text"]).strip()
        expected = str(row["category"]).strip()

        # Run actual model prediction
        predicted = str(pipeline.predict([text])[0])

        is_correct = (predicted.lower() == expected.lower())
        result_str = "PASS" if is_correct else "FAIL"

        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

        y_true.append(expected)
        y_pred.append(predicted)

        # Truncate text for clean terminal tabular display
        truncated_text = (text[:40] + "..") if len(text) > 42 else text
        print(f"{idx + 1:<3} | {expected:<12} | {predicted:<12} | {result_str:<6} | {truncated_text:<42}")

    accuracy_pct = (correct_count / total_tests) * 100

    print("-" * 80)
    print("\n[4/4] Summary & Performance Metrics:")
    print("-" * 80)
    print(f"Total Test Cases   : {total_tests}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Incorrect Cases    : {incorrect_count}")
    print(f"Overall Accuracy   : {accuracy_pct:.2f}%")
    print("-" * 80)

    # Classification Report
    labels = sorted(list(set(y_true)))
    print("\nCategory-wise Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:")
    print(f"{'Actual \\ Predicted':<18} " + " ".join([f"{lbl[:6]:>8}" for lbl in labels]))
    for i, actual_lbl in enumerate(labels):
        row_str = " ".join([f"{cm[i][j]:>8}" for j in range(len(labels))])
        print(f"{actual_lbl:<18} {row_str}")

    print("\n" + "=" * 80)
    print(" Evaluation Completed Successfully!")
    print(" Note: Accuracy reflects performance on this dedicated test dataset.")
    print(" Real-world performance may vary based on text quality, length, and vocabulary.")
    print("=" * 80)


if __name__ == "__main__":
    evaluate()
