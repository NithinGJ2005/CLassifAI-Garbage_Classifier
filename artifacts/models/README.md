# Model Artifacts Directory (`artifacts/models/`)

This directory houses the deep learning model weights and metadata for the **ClassifAI** Garbage Classifier application.

---

## 📁 File Structure & Formats

```
artifacts/models/
├── garbage_classifier.keras       # [Preferred] Retrained MobileNetV2 in Keras v3 format
├── garbage_classifier.h5          # [Fallback] Legacy MobileNetV2 model in HDF5 format
├── class_names.json               # [Optional] Authoritative class mapping
├── test_metrics.json              # [Optional] Held-out test evaluation results
└── README.md                      # This documentation
```

---

## 🔄 Model Resolution Hierarchy

When the Streamlit application starts (`app.py`), it automatically searches for models in this order:

1. **`artifacts/models/garbage_classifier.keras`**
   - Modern Keras format (TensorFlow 2.15+ / Keras 3+).
   - Produced by the Google Colab retraining workflow on `TrashBusters/combined`.
   - Selected automatically if present.
   - Status badge in UI: `Model: MobileNetV2 (.keras)`.

2. **`artifacts/models/garbage_classifier.h5`**
   - Legacy HDF5 format.
   - Used automatically as a fallback if `garbage_classifier.keras` is not found.
   - Status badge in UI: `Model: MobileNetV2 Legacy (.h5)`.

3. **Fallback Message**
   - If neither model file is found, the app displays a clear user notification and instructions on how to train or supply a model file.

---

## 🏷️ Class Names & Taxonomy

The model predicts six waste classes in the following exact order:

```json
[
  "cardboard",
  "glass",
  "metal",
  "paper",
  "plastic",
  "trash"
]
```

If `artifacts/models/class_names.json` is present, the app reads and validates the class list from this file; otherwise, it falls back to the default six classes.

---

## 📊 Evaluation Metrics (`test_metrics.json`)

When training is completed in Google Colab, `test_metrics.json` is generated containing the held-out test split performance:
- Overall accuracy & balanced accuracy
- Test loss
- Macro precision, recall, and F1-score
- Weighted F1-score
- Per-class metrics
- Average inference latency

Placing this file in `artifacts/models/` allows future dashboards or admin tools to display model evaluation metrics.

---

## ↩️ Rollback Instructions

To roll back from a newly trained `.keras` model to the legacy `.h5` baseline:
1. Rename or delete `artifacts/models/garbage_classifier.keras`:
   ```bash
   ren artifacts\models\garbage_classifier.keras garbage_classifier.keras.bak
   ```
2. Refresh or restart the Streamlit application. The app will immediately load `garbage_classifier.h5`.
