"""
classifier_utils.py
Core utilities for text validation, model loading, and prediction inference.
Designed for simple integration with Streamlit and automated evaluation.
"""

import os
from typing import Any, Dict, Optional, Tuple
import joblib

# Constants
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "classifier_pipeline.joblib")
MAX_CHAR_LIMIT = 5000
CONFIDENCE_WARNING_THRESHOLD = 0.45

# Metadata for supported categories
SUPPORTED_CATEGORIES: Dict[str, Dict[str, str]] = {
    "Technology": {
        "icon": "💻",
        "color": "#2563EB",
        "description": "Software, artificial intelligence, cybersecurity, cloud computing, and hardware.",
    },
    "Education": {
        "icon": "🎓",
        "color": "#D97706",
        "description": "Schools, academic degrees, pedagogy, curricula, libraries, and student learning.",
    },
    "Sports": {
        "icon": "⚽",
        "color": "#059669",
        "description": "Athletic competitions, tournament matches, gameplay strategies, and team sports.",
    },
    "Health": {
        "icon": "🏥",
        "color": "#DC2626",
        "description": "Medicine, clinical trials, mental wellness, physical therapy, and nutrition.",
    },
}

# Curated demonstration samples
SAMPLE_INPUTS: Dict[str, str] = {
    "Technology Sample 1": "Engineers deployed an automated CI/CD pipeline using Docker containers and Kubernetes clusters.",
    "Technology Sample 2": "Cybersecurity teams implemented multi-factor authentication to safeguard enterprise cloud databases.",
    "Education Sample 1": "University professors published peer-reviewed research papers and mentored graduate students.",
    "Education Sample 2": "High school teachers developed an interactive STEM curriculum to improve algebra test scores.",
    "Sports Sample 1": "The striker scored a spectacular volley in stoppage time to win the championship final.",
    "Sports Sample 2": "Marathon runners trained on steep hills to enhance cardiovascular endurance before the race.",
    "Health Sample 1": "Clinical trials demonstrated that daily aerobic exercise significantly reduces arterial hypertension.",
    "Health Sample 2": "Pediatricians advise administering routine seasonal immunizations to bolster childhood immunity.",
    "Ambiguous / Mixed Topic": "The sports analytics startup developed a machine learning platform to track athletic injuries.",
}


def validate_input_text(text: Any) -> Tuple[bool, str, Optional[str]]:
    """
    Validates user text input against standard criteria:
    - Must be a Python string.
    - Cannot be empty or contain only whitespace.
    - Cannot exceed the maximum character limit (5,000 characters).

    Returns:
        (is_valid, cleaned_text, error_message)
    """
    if not isinstance(text, str):
        return False, "", "Invalid input type. Expected a text string."

    cleaned = text.strip()
    if not cleaned:
        return False, "", "Input text cannot be empty or contain only whitespace."

    if len(text) > MAX_CHAR_LIMIT:
        return (
            False,
            "",
            f"Input text exceeds the maximum allowed limit of {MAX_CHAR_LIMIT:,} characters. "
            f"(Current length: {len(text):,} characters).",
        )

    return True, cleaned, None


def load_classifier(model_path: str = DEFAULT_MODEL_PATH) -> Tuple[Optional[Any], Optional[str]]:
    """
    Loads the trained scikit-learn pipeline from disk.

    Returns:
        (pipeline, error_message)
    """
    if not os.path.exists(model_path):
        return (
            None,
            f"Model file not found at '{model_path}'. "
            "Please train the model first by running 'python train_model.py'.",
        )

    try:
        pipeline = joblib.load(model_path)
        return pipeline, None
    except Exception as e:
        return (
            None,
            f"Failed to load the classifier model: {str(e)}. "
            "The model file might be corrupted. Try regenerating it with 'python train_model.py'.",
        )


def predict_text(
    text: Any,
    pipeline: Any,
    confidence_threshold: float = CONFIDENCE_WARNING_THRESHOLD,
) -> Dict[str, Any]:
    """
    Predicts the category of a given text using the provided classifier pipeline.
    Includes input validation, confidence calculation, and warning flags.

    Returns:
        Dict with keys: success, predicted_category, confidence, probabilities,
        is_low_confidence, warning, explanation, error
    """
    # 1. Input Validation
    is_valid, cleaned_text, val_error = validate_input_text(text)
    if not is_valid:
        return {
            "success": False,
            "error": val_error,
        }

    # 2. Pipeline Existence Check
    if pipeline is None:
        return {
            "success": False,
            "error": "No model loaded. Please initialize or train the model first.",
        }

    # 3. Model Inference
    try:
        predicted_category = str(pipeline.predict([cleaned_text])[0])
        probabilities_array = pipeline.predict_proba([cleaned_text])[0]
        classes = list(pipeline.classes_)

        # Map classes to their respective probabilities
        probabilities = {
            cls: float(prob) for cls, prob in zip(classes, probabilities_array)
        }
        # Sort probabilities descending
        probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

        top_confidence = float(probabilities[predicted_category])
        is_low_confidence = top_confidence < confidence_threshold

        warning = None
        if is_low_confidence:
            warning = (
                f"Low Confidence Warning: The model's top confidence score is {top_confidence:.1%}, "
                f"which is below the {confidence_threshold:.0%} confidence threshold. "
                "The text may be ambiguous, multi-topic, or contain vocabulary outside the training distribution."
            )

        explanation = (
            f"The model extracted n-gram word features using TF-IDF vectorization and evaluated them "
            f"against the trained Logistic Regression weights. '{predicted_category}' achieved the highest probability "
            f"at {top_confidence:.1%}."
        )

        return {
            "success": True,
            "predicted_category": predicted_category,
            "confidence": top_confidence,
            "probabilities": probabilities,
            "is_low_confidence": is_low_confidence,
            "warning": warning,
            "explanation": explanation,
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"An error occurred during text classification: {str(exc)}",
        }
