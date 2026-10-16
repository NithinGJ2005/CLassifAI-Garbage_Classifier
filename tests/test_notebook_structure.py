import unittest
import json
import os
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/ClassifAI_MobileNetV2_Colab.ipynb")

class TestNotebookStructure(unittest.TestCase):

    def test_notebook_file_exists(self):
        self.assertTrue(NOTEBOOK_PATH.exists(), "Notebook file not found at expected path")

    def test_notebook_valid_json(self):
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("cells", data)
        self.assertIn("metadata", data)
        self.assertEqual(data.get("nbformat"), 4)

    def test_notebook_accelerator_metadata(self):
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        self.assertEqual(meta.get("accelerator"), "GPU", "Notebook should specify GPU accelerator metadata")

    def test_notebook_contains_key_steps(self):
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_code_sources = []
        for cell in data["cells"]:
            if cell.get("cell_type") == "code":
                all_code_sources.append("".join(cell.get("source", [])))

        full_code = "\n".join(all_code_sources)

        # Check required components in notebook code
        self.assertIn("TrashBusters/combined", full_code, "Must reference the dataset")
        self.assertIn("MobileNetV2", full_code, "Must reference MobileNetV2 architecture")
        self.assertIn("base_model.trainable = False", full_code, "Phase 1 must freeze backbone")
        self.assertIn("BatchNormalization", full_code, "Must explicitly handle BatchNormalization freezing")
        self.assertIn("1e-5", full_code, "Phase 2 must use fine-tuning learning rate (1e-5)")
        self.assertIn("confusion_matrix", full_code, "Must generate confusion matrix")
        self.assertIn("classification_report", full_code, "Must generate classification report")
        self.assertIn("garbage_classifier.keras", full_code, "Must save garbage_classifier.keras")
        self.assertIn("classifai_trained_artifacts", full_code, "Must package artifacts into zip")

if __name__ == "__main__":
    unittest.main()
