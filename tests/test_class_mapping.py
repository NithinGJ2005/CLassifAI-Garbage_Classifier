import unittest
import json
import tempfile
from pathlib import Path

# Required 6-class order
EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

SDG_MAPPING = {
    "plastic": [6, 12, 14],
    "paper": [12, 13, 15],
    "glass": [12, 13],
    "metal": [7, 12, 13],
    "cardboard": [12, 15],
    "trash": [7, 12],
}

IMPACT_MAPPING = {
    "cardboard": [
        "📦 Cardboard Waste:",
        "• Virgin cardboard: emits approx. 0.5–0.9 kg CO₂/kg.",
        "• Recycled cardboard: 0.2–0.4 kg CO₂/kg.",
        "♻ Recycling reduces water, energy use, and deforestation."
    ],
    "plastic": [
        "🧴 Plastic Waste:",
        "• PET (bottles): 2.1–3.5 kg CO₂/kg.",
        "• HDPE (containers): 1.7–2.8 kg CO₂/kg.",
        "• General plastic avg: 3–6 kg CO₂/kg.",
        "♻ Recycling plastic saves up to 30–60% emissions."
    ],
    "glass": [
        "🍾 Glass Waste:",
        "• New glass: 0.6–1.2 kg CO₂/kg due to melting.",
        "• Recycled glass: 0.2–0.4 kg CO₂/kg.",
        "♻ Glass is 100% recyclable without quality loss."
    ],
    "metal": [
        "🛠 Metal Waste:",
        "• Aluminum: 10–13 kg CO₂/kg (virgin).",
        "• Aluminum (Recycled): 0.6–1 kg CO₂/kg.",
        "• Steel: 1.8–2.5 kg CO₂/kg (60% less if recycled).",
        "♻ Recycling metals saves significant energy."
    ],
    "paper": [
        "📄 Paper Waste:",
        "• Virgin paper: 1.8–2.5 kg CO₂/kg.",
        "• Recycled paper: 0.9–1.5 kg CO₂/kg.",
        "♻ Recycling paper reduces landfill methane & preserves forests."
    ],
    "trash": [
        "🗑 Mixed General Trash:",
        "• Emissions vary widely depending on materials.",
        "• Can exceed 5 kg CO₂/kg (esp. with food/plastic).",
        "♻ Waste separation reduces footprint significantly."
    ]
}

def load_class_names_from_file(file_path):
    p = Path(file_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, list) and len(names) == 6:
                return [n.lower().strip() for n in names]
        except Exception:
            pass
    return EXPECTED_CLASSES

class TestClassMapping(unittest.TestCase):

    def test_class_names_count_and_order(self):
        self.assertEqual(len(EXPECTED_CLASSES), 6)
        self.assertEqual(EXPECTED_CLASSES[0], "cardboard")
        self.assertEqual(EXPECTED_CLASSES[1], "glass")
        self.assertEqual(EXPECTED_CLASSES[2], "metal")
        self.assertEqual(EXPECTED_CLASSES[3], "paper")
        self.assertEqual(EXPECTED_CLASSES[4], "plastic")
        self.assertEqual(EXPECTED_CLASSES[5], "trash")

    def test_class_names_are_unique(self):
        self.assertEqual(len(EXPECTED_CLASSES), len(set(EXPECTED_CLASSES)))

    def test_sdg_mapping_completeness(self):
        # All 6 classes must be mapped to valid SDGs
        for cls in EXPECTED_CLASSES:
            self.assertIn(cls, SDG_MAPPING, f"Class '{cls}' missing from SDG_MAPPING")
            sdgs = SDG_MAPPING[cls]
            self.assertIsInstance(sdgs, list)
            self.assertGreater(len(sdgs), 0)
            for sdg_num in sdgs:
                self.assertIn(sdg_num, range(1, 18), f"Invalid SDG number {sdg_num}")

    def test_impact_mapping_completeness(self):
        # All 6 classes must have environmental insights
        for cls in EXPECTED_CLASSES:
            self.assertIn(cls, IMPACT_MAPPING, f"Class '{cls}' missing from IMPACT_MAPPING")
            insights = IMPACT_MAPPING[cls]
            self.assertIsInstance(insights, list)
            self.assertGreaterEqual(len(insights), 3)

    def test_load_class_names_custom_json(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf:
            json.dump(["cardboard", "glass", "metal", "paper", "plastic", "trash"], tf)
            temp_name = tf.name

        loaded = load_class_names_from_file(temp_name)
        self.assertEqual(loaded, EXPECTED_CLASSES)

    def test_load_class_names_fallback_on_corrupt(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf:
            tf.write("{bad json content}")
            temp_name = tf.name

        loaded = load_class_names_from_file(temp_name)
        self.assertEqual(loaded, EXPECTED_CLASSES)

if __name__ == "__main__":
    unittest.main()
