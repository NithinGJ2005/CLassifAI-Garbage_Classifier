# 🚀 Training ClassifAI in Google Colab

This guide provides end-to-end instructions for retraining, evaluating, exporting, and integrating the updated **MobileNetV2** model for **ClassifAI** using Google Colab and the **TrashBusters Combined Dataset**.

---

## 📋 Table of Contents
1. [Dataset Overview](#1-dataset-overview)
2. [Google Colab Setup & Runtime Configuration](#2-google-colab-setup--runtime-configuration)
3. [Step-by-Step Training Workflow](#3-step-by-step-training-workflow)
   - [Phase 1: Classifier Head Training](#phase-1-classifier-head-training-frozen-backbone)
   - [Phase 2: MobileNetV2 Fine-Tuning](#phase-2-mobilenetv2-fine-tuning-top-30-layers)
4. [Evaluation & Metric Verification](#4-evaluation--metric-verification)
5. [Downloading & Organizing Deliverables](#5-downloading--organizing-deliverables)
6. [Local Integration into Streamlit](#6-local-integration-into-streamlit)
7. [Rollback & Compatibility Safeguards](#7-rollback--compatibility-safeguards)
8. [Troubleshooting & FAQs](#8-troubleshooting--faqs)

---

## 1. Dataset Overview

The retraining pipeline uses the official **Waste Classification Harmonized Dataset**:
- **Hugging Face Hub:** [`TrashBusters/combined`](https://huggingface.co/datasets/TrashBusters/combined)
- **License:** MIT
- **Total Images:** 29,849 (~2.07 GB)
- **Official Pre-defined Splits:**
  - `train`: 20,894 images
  - `validation`: 4,477 images
  - `test`: 4,478 images
- **Target Taxonomy (Strict Order Preserved):**
  1. `0: cardboard`
  2. `1: glass`
  3. `2: metal`
  4. `3: paper`
  5. `4: plastic`
  6. `5: trash`

> [!IMPORTANT]
> The dataset splits (`train`, `validation`, `test`) must be preserved as provided. The test split is held out strictly for final evaluation and must **never** be used during training or hyperparameter tuning.

---

## 2. Google Colab Setup & Runtime Configuration

### Step 2.1: Open the Notebook in Google Colab
1. Open your web browser and go to [Google Colab](https://colab.research.google.com/).
2. In the modal dialog, select the **Upload** tab.
3. Browse and select the notebook from this repository:
   ```
   notebooks/ClassifAI_MobileNetV2_Colab.ipynb
   ```
   *(Alternatively, if your repository is pushed to GitHub, use the **GitHub** tab in Colab to search and open it directly).*

### Step 2.2: Select a GPU Hardware Accelerator
MobileNetV2 training requires a GPU for fast training:
1. In Colab's top menu, click **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (available on free tier).
3. Under **GPU type**, ensure **Standard** is selected.
4. Click **Save**.

---

## 3. Step-by-Step Training Workflow

Execute each cell in the notebook sequentially from top to bottom.

### Step 1: Environment Check & Dependencies
- Cell 1.1 checks Python and TensorFlow versions and executes `nvidia-smi` to confirm GPU availability.
- Cell 1.2 installs required libraries:
  ```bash
  !pip install -q datasets scikit-learn matplotlib pandas pillow seaborn
  ```
- Cell 1.3 sets random seeds (`42`) across Python, NumPy, and TensorFlow to ensure reproducible results.

### Step 2: Dataset Loading & Inspection
- Cell 2.1 streams and loads `TrashBusters/combined` directly from Hugging Face into Colab cache.
- Cell 2.2 verifies that label IDs `0..5` correspond strictly to `["cardboard", "glass", "metal", "paper", "plastic", "trash"]`. It saves `class_names.json`.
- Cell 2.3 computes the exact class distributions across splits and plots a comparison bar chart.
- Cell 2.4 renders a sample 2×3 visual grid displaying sample images from all six waste categories.
- Cell 2.5 performs data sanity checks (verifying RGB color mode, dimensions) and computes MD5 hash signatures to check for potential cross-split duplicate leakage.

### Step 3: Data Pipeline & Training Data Augmentation
- Target input resolution: **224 × 224 pixels** with 3 RGB channels.
- Normalization: Preprocessed via `tensorflow.keras.applications.mobilenet_v2.preprocess_input` (scaling pixels to $[-1, 1]$).
- Augmentation (training split only):
  - Random horizontal flipping
  - Small rotations ($\pm 8\% \approx 15^\circ$)
  - Moderate zoom ($\pm 10\%$)
  - Translation ($\pm 5\%$)
  - Contrast variation ($\pm 10\%$)
- Memory Safety: The notebook streams data using an Arrow-backed generator with batch slicing and `AUTOTUNE` prefetching. Total memory usage remains below 200 MB, preventing Colab Out-Of-Memory (OOM) crashes.

### Phase 1: Classifier Head Training (Frozen Backbone)
- Backbone: ImageNet-pretrained `MobileNetV2` frozen (`base_model.trainable = False`).
- Classifier Head:
  - `GlobalAveragePooling2D()`
  - `Dense(256, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(6, activation='softmax')`
- Optimizer: `Adam(learning_rate=0.001)`
- Loss: `SparseCategoricalCrossentropy()`
- Callbacks:
  - `EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)`
  - `ModelCheckpoint('colab_artifacts/phase1_best_model.keras', save_best_only=True)`
- Max Epochs: 20 (typically converges in 8–14 epochs).

### Phase 2: MobileNetV2 Fine-Tuning (Top 30 Layers)
- The best weights from Phase 1 are restored.
- The top 30 layers of MobileNetV2 are unfrozen.
- **Batch Normalization layers remain strictly frozen** (`layer.trainable = False` for all `BatchNormalization` instances) to prevent mean and variance drift on fine-tuning batches.
- Optimizer: Recompiled with `Adam(learning_rate=1e-5)` (100x lower rate for stability).
- Max Epochs: 15.
- Callbacks: EarlyStopping (patience=3), ReduceLROnPlateau, ModelCheckpoint.
- **Model Selection:** The notebook computes Validation Macro-F1 and Validation Loss for both Phase 1 and Phase 2. The winning candidate is selected and saved as `garbage_classifier.keras`.

---

## 4. Evaluation & Metric Verification

Once the best candidate model is selected, Step 6 evaluates it against the **Held-Out Test Split** (`4,478 images`).

The notebook automatically generates:
1. **Summary Metrics (`test_metrics.json`):**
   - Overall Test Accuracy
   - Balanced Accuracy
   - Macro Precision, Macro Recall, Macro F1-score
   - Weighted F1-score
   - Test Loss
   - Average Inference Latency (ms per image)
2. **Per-Class Breakdown (`classification_report.csv`):**
   - Precision, Recall, and F1 for `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
3. **Confusion Matrix (`confusion_matrix.png`):**
   - Side-by-side raw counts and normalized recall percentage heatmaps.
4. **Learning Curves (`training_curves.png`):**
   - Stitched Phase 1 and Phase 2 loss and accuracy curves.
5. **Misclassified Image Inspection:**
   - Visual grid of misclassified test images with true label vs predicted label and confidence.
6. **Generalization Analysis:**
   - Accuracy breakdown across the constituent datasets (`source_dataset`).
7. **Optional Baseline Comparison:**
   - If you upload the legacy `garbage_classifier.h5` into Colab root, the notebook runs a head-to-head comparison on the exact same test images.

---

## 5. Downloading & Organizing Deliverables

Cell 7.1 packages all training outputs into a zip archive:
`classifai_trained_artifacts.zip`

Cell 7.2 triggers an automatic browser download.

### Contents of `classifai_trained_artifacts.zip`:
```
classifai_trained_artifacts.zip
├── garbage_classifier.keras       # The trained MobileNetV2 model (Keras v3 format)
├── class_names.json               # Ordered list of classes
├── test_metrics.json              # Full evaluation metrics dictionary
├── classification_report.csv      # Per-class classification report
├── confusion_matrix.png           # Confusion matrix visualization
├── training_curves.png            # Loss and accuracy curves
├── training_history.csv           # Detailed epoch history log
└── evaluation_summary.md          # Markdown executive summary
```

> [!TIP]
> If the browser blocks automatic downloads, you can also download the file manually:
> Click the **Folder icon (Files)** on the left sidebar in Google Colab, locate `classifai_trained_artifacts.zip`, click the three dots (`⋮`), and select **Download**.

---

## 6. Local Integration into Streamlit

### Step 6.1: Extract Artifacts into the Repository
1. Unzip `classifai_trained_artifacts.zip` on your local computer.
2. Copy the production model and metadata into `artifacts/models/`:
   ```bash
   # Target directory: artifacts/models/
   artifacts/models/garbage_classifier.keras   <-- Place new model here
   artifacts/models/class_names.json           <-- Place class names here
   artifacts/models/test_metrics.json          <-- Place metrics here
   ```
3. Keep the existing `artifacts/models/garbage_classifier.h5` in place as a fallback.

### Step 6.2: Run Unit Tests
Verify that the model, preprocessing, class order, and threshold logic match expectations:
```bash
python -m unittest discover tests
```
All tests should pass without errors.

### Step 6.3: Launch Streamlit App Locally
```bash
streamlit run app.py
```
The app will automatically:
- Detect and load `artifacts/models/garbage_classifier.keras`.
- Display a badge: **"Model: MobileNetV2 (.keras)"**.
- Classify uploaded images with confidence scores.
- Trigger a warning if prediction confidence is below the threshold (default: 65%).
- Display relevant SDG icons and carbon footprint insights.

---

## 7. Rollback & Compatibility Safeguards

ClassifAI includes automatic backward compatibility:
- If `garbage_classifier.keras` is removed or renamed, `app.py` automatically falls back to `garbage_classifier.h5`.
- If neither model file is present, the app renders a clean warning message instructing the user where to place the model file, preventing application crashes.

To roll back to the legacy model at any time:
```bash
# Rename the .keras model or delete it
ren artifacts\models\garbage_classifier.keras garbage_classifier.keras.bak
```
Refresh the Streamlit app in your browser; it will seamlessly load `garbage_classifier.h5`.

---

## 8. Troubleshooting & FAQs

### Q: Colab disconnected before training finished. What should I do?
Colab free tier may time out if the tab is closed or idle. Keep the browser tab open. If disconnected, simply re-run the cells—Hugging Face dataset caching ensures re-downloads are instantaneous.

### Q: What if GPU memory runs out (OOM error)?
In Cell 1.3 or Cell 3.2, change `BATCH_SIZE = 32` to `BATCH_SIZE = 16`. MobileNetV2 uses very little VRAM (typically < 3 GB at batch size 32), so OOM is rare on a T4 GPU.

### Q: Why keep Batch Normalization layers frozen during fine-tuning?
When fine-tuning on a small or pre-trained dataset with small batch sizes, updating Batch Normalization moving statistics can cause destabilization and catastrophic forgetting. Keeping BN frozen ensures stable feature representations.

### Q: Are pixel values preprocessed twice?
No. Images are converted to RGB, resized to $224 \times 224$, and normalized strictly once using `mobilenet_v2.preprocess_input` (mapping values to $[-1, 1]$) in both Colab and `app.py`.
