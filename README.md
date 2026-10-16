# ClassifAI: Smart Waste Classification System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://classif-ai-a-garbage-classifier-app.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![Tests](https://img.shields.io/badge/Tests-20%20Passed-brightgreen.svg)](#running-unit-tests)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-82.67%25-success.svg)](#model-architecture--held-out-benchmark-results)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **"Turning Trash into Data for a Greener Planet."**

---

## Overview

**ClassifAI** is a deep-learning-powered web application engineered to automate waste segregation, promote circular economy practices, and raise climate awareness. Built with **Streamlit** and **TensorFlow / Keras**, ClassifAI identifies waste materials from uploaded images and provides actionable, data-backed insights on:

* **Material Classification:** Automatically identifies 6 primary waste streams (**Cardboard**, **Glass**, **Metal**, **Paper**, **Plastic**, **Trash**).
* **Environmental Impact:** Delivers instant **Carbon Footprint** estimates (comparing virgin material extraction vs. recycled processing in $\text{kg CO}_2\text{/kg}$).
* **Sustainability Integration:** Maps classified items to relevant **United Nations Sustainable Development Goals (SDGs)** (SDG 6, 7, 12, 13, 14, 15).
* **Uncertainty Awareness:** Built-in low-confidence detection with an interactive sensitivity slider to flag ambiguous or mixed items for manual verification.

---

## Key Features

* **Real-Time Material Detection:** Upload JPG, JPEG, or PNG images to receive instant predictions.
* **MobileNetV2 Architecture:** Employs an ImageNet-pretrained MobileNetV2 backbone fine-tuned on the harmonized `TrashBusters/combined` dataset (29,849 images).
* **Model Resolution Hierarchy:** Automatically prioritizes modern `.keras` weights (`garbage_classifier.keras`) with a seamless fallback to legacy `.h5` models (`garbage_classifier.h5`).
* **Low-Confidence Warning System:** User-adjustable threshold slider (40%–90%, default 65%) that flags borderline classifications to prevent contamination in recycling streams.
* **In-App Evaluation Dashboard:** Real-time sidebar expander displaying test accuracy (82.67%), Macro F1-score (0.8264), and inference latency (~6.84 ms).
* **Eco & Climate Insights:** Calculates potential CO₂ reductions achieved through recycling each classified material.
* **SDG Visual Mapping:** Visual badges displaying alignment with United Nations Sustainable Development Goals.
* **Modern High-Contrast UI:** Full dark mode with a responsive three-column layout, custom cards, and styled file uploader.

---

## Interface Preview

| **Upload & Classify Interface** | **Results & Eco-Insights** |
|:-------------------------------:|:--------------------------:|
| ![Upload Screen](assets/Screenshot%202026-01-31%20183621.png) | ![Results Screen](assets/Screenshot%202026-01-31%20183720.png) |

---

## Model Architecture & Held-Out Benchmark Results

The model was retrained on the **Waste Classification Harmonized Dataset** ([`TrashBusters/combined`](https://huggingface.co/datasets/TrashBusters/combined)) using a two-phase transfer learning and fine-tuning strategy:
1. **Phase 1 (Feature Extraction):** MobileNetV2 backbone frozen; trained custom Dense(256) + Dropout(0.5) + Softmax(6) head with Adam ($\text{lr}=10^{-3}$).
2. **Phase 2 (Fine-Tuning):** Unfroze the top 30 layers with frozen BatchNormalization layers using low learning rate Adam ($\text{lr}=10^{-5}$).

### Overall Test Split Performance (4,478 Unseen Images)

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **82.67%** |
| **Balanced Accuracy** | **82.70%** |
| **Macro Precision** | **0.8267** |
| **Macro Recall** | **0.8270** |
| **Macro F1-Score** | **0.8264** |
| **Weighted F1-Score** | **0.8262** |
| **Test Loss** | **0.5003** |
| **Average Inference Latency** | **6.84 ms / image** |

### Per-Class Evaluation Breakdown

| Waste Class | Precision | Recall | F1-Score | Held-Out Support |
| :--- | :---: | :---: | :---: | :---: |
| **Cardboard** | 86.11% | 88.60% | **0.8734** | 728 images |
| **Glass** | 86.68% | 85.07% | **0.8587** | 750 images |
| **Metal** | 80.51% | 87.60% | **0.8391** | 750 images |
| **Paper** | 79.35% | 78.40% | **0.7887** | 750 images |
| **Plastic** | 78.86% | 73.60% | **0.7614** | 750 images |
| **Trash (General)** | 84.51% | 82.93% | **0.8371** | 750 images |

Evaluation plots, including the [Confusion Matrix](artifacts/models/confusion_matrix.png), [Training Curves](artifacts/models/training_curves.png), and [Class Distribution](artifacts/models/class_distribution.png), are saved in `artifacts/models/`.

---

## Tech Stack

| Domain | Technology | Description |
| :--- | :--- | :--- |
| **Frontend UI** | [Streamlit](https://streamlit.io/) | High-performance reactive web application framework |
| **Deep Learning** | [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/) | Model architecture, inference engine, and training pipeline |
| **Computer Vision** | [Pillow (PIL)](https://python-pillow.org/) & [NumPy](https://numpy.org/) | Preprocessing, color-space management, and tensor transformations |
| **Dataset & Hub** | [Hugging Face Datasets](https://huggingface.co/datasets/TrashBusters/combined) | Streaming 29.8k multi-source harmonized waste images |
| **Evaluation** | [Scikit-Learn](https://scikit-learn.org/) & [Matplotlib](https://matplotlib.org/) | Confusion matrices, classification reports, and training curves |
| **Testing** | Python `unittest` | Automated test suite validating data pipelines and model resolution |

---

## Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/NithinGJ2005/CLassifAI.git
cd CLassifAI
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
streamlit run app.py
```
The application will launch in your default browser at `http://localhost:8501`.

---

## Usage Guide

1. **Launch App:** Access the live demo or run locally with `streamlit run app.py`.
2. **Upload Waste Photo:** Click **Browse files** or drag-and-drop an image (`.jpg`, `.jpeg`, `.png`).
3. **Run Classification:** Click the blue **Classify Waste** button.
4. **Inspect Prediction:**
   - Review predicted class and statistical confidence percentage.
   - Check if a **Low Confidence Warning** appears if confidence falls below the threshold set in the sidebar.
   - Inspect the mapped **United Nations SDGs** icons.
   - Read the **Carbon Footprint Awareness** card to learn about emissions and recycling savings.
5. **Adjust Settings:** Use the sidebar slider to customize the low-confidence sensitivity (40% to 90%).

---

## Model Retraining Workflow (Google Colab)

ClassifAI provides a complete, reproducible retraining pipeline designed for free Google Colab T4 GPU environments:

* **Retraining Notebook:** [`notebooks/ClassifAI_MobileNetV2_Colab.ipynb`](notebooks/ClassifAI_MobileNetV2_Colab.ipynb)
* **Step-by-Step Retraining Guide:** [`TRAINING_IN_COLAB.md`](TRAINING_IN_COLAB.md)
* **Dataset:** [TrashBusters/combined](https://huggingface.co/datasets/TrashBusters/combined) (20,894 train, 4,477 validation, 4,478 test images)

### Running in Colab:
1. Upload `notebooks/ClassifAI_MobileNetV2_Colab.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Enable GPU: **Runtime** > **Change runtime type** > **T4 GPU**.
3. Run all cells sequentially. The notebook handles dataset download, two-phase training, validation-based model selection, held-out test evaluation, and generates export-ready artifacts into `artifacts/models/`.

---

## Running Unit Tests

A comprehensive unit test suite ensures preprocessing safety, class taxonomy preservation, fallback resolution, and notebook cell validity:

```bash
python -m unittest discover tests
```

### Test Coverage Highlights:
- **`tests/test_preprocessing.py`:** Validates image resizing to $(224, 224)$, conversion to RGB, and single-pass MobileNetV2 normalization $[-1, 1]$.
- **`tests/test_class_mapping.py`:** Enforces immutable 6-class alphabetical taxonomy and validates all SDG and Carbon Impact dictionary mappings.
- **`tests/test_model_pipeline.py`:** Tests `.keras` primary and `.h5` fallback resolution, missing model handling, and confidence threshold warnings.
- **`tests/test_notebook_structure.py`:** Validates JSON structure, required markdown anchors, code cells, and random seed consistency in the Colab notebook.

---

## Project Structure

```bash
ClassifAI/
├── .devcontainer/                          # Dev container specification
│   └── devcontainer.json
├── .streamlit/                             # Streamlit theme & layout configuration
│   └── config.toml
├── artifacts/
│   └── models/
│       ├── garbage_classifier.keras        # [Primary] Fine-tuned MobileNetV2 (Keras v3)
│       ├── garbage_classifier.h5           # [Fallback] Legacy baseline model
│       ├── phase1_best_model.keras         # Phase 1 classifier head checkpoint
│       ├── phase2_best_model.keras         # Phase 2 fine-tuned model checkpoint
│       ├── class_names.json                # Immutable 6-class order definition
│       ├── test_metrics.json               # Full held-out test evaluation metrics
│       ├── classification_report.csv       # Precision, recall, and F1 per class
│       ├── training_history.csv            # Epoch-by-epoch loss and accuracy metrics
│       ├── evaluation_summary.md           # Formatted evaluation benchmark summary
│       ├── confusion_matrix.png            # Held-out test split confusion matrix
│       ├── training_curves.png             # Loss and accuracy progression curves
│       ├── class_distribution.png          # Dataset split class distribution plot
│       └── README.md                       # Model artifacts documentation & rollback guide
├── assets/
│   ├── Screenshot 2026-01-31 183621.png    # Upload screen preview
│   ├── Screenshot 2026-01-31 183720.png    # Results screen preview
│   └── sdg/                                # United Nations SDG official badges
│       ├── sdg_6.png                       # SDG 6: Clean Water and Sanitation
│       ├── sdg_7.png                       # SDG 7: Affordable and Clean Energy
│       ├── sdg_12.png                      # SDG 12: Responsible Consumption and Production
│       ├── sdg_13.png                      # SDG 13: Climate Action
│       ├── sdg_14.png                      # SDG 14: Life Below Water
│       └── sdg_15.png                      # SDG 15: Life on Land
├── notebooks/
│   └── ClassifAI_MobileNetV2_Colab.ipynb   # Complete Colab retraining notebook
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py               # Preprocessing & normalization unit tests
│   ├── test_class_mapping.py               # Class taxonomy & SDG mapping unit tests
│   ├── test_model_pipeline.py              # Model fallback & threshold unit tests
│   └── test_notebook_structure.py          # Colab notebook cell integrity tests
├── app.py                                  # Main Streamlit web application
├── requirements.txt                        # Application runtime dependencies
├── TRAINING_IN_COLAB.md                    # Detailed Google Colab retraining guide
├── implementation_plan.md                  # Project roadmap & implementation specifications
├── LICENSE                                 # MIT License
└── README.md                               # Project documentation
```

---

## Contributing

Contributions are welcomed! If you would like to contribute:
1. **Fork the Repository:** Click the 'Fork' button on GitHub.
2. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit Your Changes:**
   ```bash
   git commit -m "Add AmazingFeature"
   ```
4. **Run Unit Tests:** Ensure all tests pass:
   ```bash
   python -m unittest discover tests
   ```
5. **Push to Branch:**
   ```bash
   git push origin feature/AmazingFeature
   ```
6. **Open a Pull Request.**

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.

---

<div align="center">
  <p>Made by <b>Nithin G J</b></p>
  <p>
    <a href="https://github.com/NithinGJ2005" target="_blank">
      <img src="https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github" alt="GitHub Profile" />
    </a>
    &nbsp;
    <a href="https://classif-ai-a-garbage-classifier-app.streamlit.app/" target="_blank">
      <img src="https://img.shields.io/badge/Live-Demo-FF4B4B?style=for-the-badge&logo=streamlit" alt="Live Demo" />
    </a>
  </p>
</div>
