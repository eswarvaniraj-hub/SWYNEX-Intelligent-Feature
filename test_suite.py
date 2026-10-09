"""
test_suite.py
Automated unit and integration tests for Smart Text Intelligence.
Validates input sanitization, boundary limits, missing model handling, and deterministic inference.
"""

import unittest
import os
import classifier_utils as cu

class TestTextClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline, cls.load_err = cu.load_classifier()

    def test_01_model_loaded_successfully(self):
        """Verify that the model pipeline loaded without errors."""
        self.assertIsNotNone(self.pipeline, f"Model failed to load: {self.load_err}")
        self.assertIsNone(self.load_err)

    def test_02_empty_input_rejected(self):
        """Verify empty string is rejected with friendly error."""
        res = cu.predict_text("", self.pipeline)
        self.assertFalse(res["success"])
        self.assertIn("cannot be empty", res["error"])

    def test_03_whitespace_only_rejected(self):
        """Verify whitespace-only string is rejected with friendly error."""
        res = cu.predict_text("   \n\t  \r  ", self.pipeline)
        self.assertFalse(res["success"])
        self.assertIn("cannot be empty", res["error"])

    def test_04_oversized_text_rejected(self):
        """Verify inputs exceeding 5,000 characters are rejected."""
        long_text = "word " * 1200  # > 5,000 chars
        self.assertGreater(len(long_text), 5000)
        res = cu.predict_text(long_text, self.pipeline)
        self.assertFalse(res["success"])
        self.assertIn("exceeds the maximum allowed limit", res["error"])

    def test_05_non_string_types_rejected(self):
        """Verify non-string types return graceful validation error."""
        for invalid_input in [None, 12345, 3.1415, ["sample", "text"], {"key": "val"}]:
            res = cu.predict_text(invalid_input, self.pipeline)
            self.assertFalse(res["success"])
            self.assertIn("Expected a text string", res["error"])

    def test_06_missing_model_handled_gracefully(self):
        """Verify missing model file yields friendly error message."""
        pipe, err = cu.load_classifier("models/non_existent_file.joblib")
        self.assertIsNone(pipe)
        self.assertIn("Model file not found", err)
        self.assertIn("python train_model.py", err)

        res = cu.predict_text("Valid test text", None)
        self.assertFalse(res["success"])
        self.assertIn("No model loaded", res["error"])

    def test_07_valid_predictions_structure(self):
        """Verify valid prediction returns all required fields and categories."""
        text = "Cardiovascular workouts enhance cardiac health and lower resting blood pressure."
        res = cu.predict_text(text, self.pipeline)

        self.assertTrue(res["success"])
        self.assertEqual(res["predicted_category"], "Health")
        self.assertIn("confidence", res)
        self.assertGreater(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertIn("probabilities", res)
        self.assertEqual(len(res["probabilities"]), 4)
        for cat in ["Technology", "Education", "Sports", "Health"]:
            self.assertIn(cat, res["probabilities"])
        self.assertIn("explanation", res)

    def test_08_low_confidence_flagging(self):
        """Verify ambiguous text flags low confidence."""
        ambiguous_text = "The sports analytics startup developed a machine learning platform to track athletic injuries."
        res = cu.predict_text(ambiguous_text, self.pipeline)
        self.assertTrue(res["success"])
        self.assertTrue(res["is_low_confidence"])
        self.assertIsNotNone(res["warning"])
        self.assertIn("Low Confidence Warning", res["warning"])

    def test_09_deterministic_inference(self):
        """Verify repeated predictions produce identical probabilities."""
        text = "Deep neural networks are trained with gradient descent optimization."
        res1 = cu.predict_text(text, self.pipeline)
        res2 = cu.predict_text(text, self.pipeline)
        self.assertEqual(res1["predicted_category"], res2["predicted_category"])
        self.assertEqual(res1["confidence"], res2["confidence"])
        self.assertEqual(res1["probabilities"], res2["probabilities"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
