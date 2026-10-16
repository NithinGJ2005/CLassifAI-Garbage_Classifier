import unittest
import numpy as np
from pathlib import Path

def resolve_model_path(keras_path, h5_path):
    if keras_path.exists():
        return keras_path, "MobileNetV2 (.keras)"
    elif h5_path.exists():
        return h5_path, "MobileNetV2 Legacy (.h5)"
    return None, "Missing Model"

def evaluate_confidence(probs, threshold=65.0):
    idx = int(np.argmax(probs))
    conf = float(probs[idx] * 100)
    is_low = conf < threshold
    return idx, conf, is_low

class TestModelPipeline(unittest.TestCase):

    def test_model_resolution_prefers_keras(self):
        # When both paths exist, keras is chosen
        class MockPath:
            def __init__(self, exists_val, name="mock"):
                self._exists = exists_val
                self.name = name
            def exists(self):
                return self._exists

        keras_p = MockPath(True, "garbage_classifier.keras")
        h5_p = MockPath(True, "garbage_classifier.h5")
        chosen, m_type = resolve_model_path(keras_p, h5_p)
        self.assertEqual(chosen.name, "garbage_classifier.keras")
        self.assertEqual(m_type, "MobileNetV2 (.keras)")

    def test_model_resolution_falls_back_to_h5(self):
        class MockPath:
            def __init__(self, exists_val, name="mock"):
                self._exists = exists_val
                self.name = name
            def exists(self):
                return self._exists

        keras_p = MockPath(False, "garbage_classifier.keras")
        h5_p = MockPath(True, "garbage_classifier.h5")
        chosen, m_type = resolve_model_path(keras_p, h5_p)
        self.assertEqual(chosen.name, "garbage_classifier.h5")
        self.assertEqual(m_type, "MobileNetV2 Legacy (.h5)")

    def test_model_resolution_handles_missing_files(self):
        class MockPath:
            def exists(self):
                return False

        chosen, m_type = resolve_model_path(MockPath(), MockPath())
        self.assertIsNone(chosen)
        self.assertEqual(m_type, "Missing Model")

    def test_confidence_threshold_logic(self):
        # Case 1: High confidence prediction (85%)
        probs_high = np.array([0.02, 0.85, 0.03, 0.04, 0.05, 0.01])
        idx, conf, is_low = evaluate_confidence(probs_high, threshold=65.0)
        self.assertEqual(idx, 1)  # glass
        self.assertAlmostEqual(conf, 85.0, places=2)
        self.assertFalse(is_low)

        # Case 2: Low confidence prediction (45%)
        probs_low = np.array([0.15, 0.10, 0.12, 0.45, 0.10, 0.08])
        idx, conf, is_low = evaluate_confidence(probs_low, threshold=65.0)
        self.assertEqual(idx, 3)  # paper
        self.assertAlmostEqual(conf, 45.0, places=2)
        self.assertTrue(is_low)

    def test_softmax_invariants(self):
        # Mock 6-class softmax output
        raw_logits = np.array([2.1, 0.5, 4.3, 1.2, 0.8, -0.5])
        exp_vals = np.exp(raw_logits - np.max(raw_logits))
        softmax_probs = exp_vals / np.sum(exp_vals)

        self.assertEqual(len(softmax_probs), 6)
        self.assertAlmostEqual(float(np.sum(softmax_probs)), 1.0, places=5)
        self.assertTrue(np.all(softmax_probs >= 0.0) and np.all(softmax_probs <= 1.0))

if __name__ == "__main__":
    unittest.main()
