# ♻️ ClassifAI: Smart Waste Classification System

**ClassifAI** is a Deep Learning-powered web application designed to promote sustainable waste management. It accurately classifies garbage into 6 categories and provides users with immediate feedback on the environmental impact (Carbon Footprint) and related Sustainable Development Goals (SDGs).

## 🌟 Features
* **AI-Powered Classification:** Uses a fine-tuned **MobileNetV2** model to identify Cardboard, Glass, Metal, Paper, Plastic, and Trash.
* **Carbon Footprint Awareness:** Displays estimated CO₂ emission data for each material type.
* **SDG Integration:** Maps waste types to relevant UN Sustainable Development Goals (e.g., SDG 12: Responsible Consumption).
* **Interactive UI:** Built with **Streamlit**, featuring a dark-mode interface with high-contrast visualization.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Python)
* **Backend/Model:** TensorFlow, Keras (MobileNetV2)
* **Image Processing:** PIL (Python Imaging Library), NumPy

## 📂 Dataset
The model was trained on the [Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification), consisting of 2,527 images across 6 classes.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/garbage-classifier-app.git](https://github.com/YOUR_USERNAME/garbage-classifier-app.git)
   cd garbage-classifier-app
