# Plant Disease Detection AI 🌿

This project uses Deep Learning (MobileNetV2) to detect plant diseases from leaf images. It includes a training notebook and a modern Streamlit web application.

## 🚀 Quick Start

### 1. Setup Environment
Open your terminal in this folder (`Plant_Disease_Detection`) and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Get the Dataset & Train Model
This project is configured to train a **Tomato Disease Detection Model** on a subset of the PlantDoc dataset.

1.  Ensure the **dataset** folder is present in this directory (specifically `dataset/train`).
2.  Run the training script:
```bash
python3 main.py
```
This will train a Random Forest model on just the Tomato images and save `best_tomato_model.pkl` and `tomato_class_names.json`.

### 3. Run the App
Launch the interactive web interface:
```bash
streamlit run app.py
```

## ✨ Features
- **Dual Input**: Upload images or use your webcam.
- **Smart Analysis**: Uses MobileNetV2 for fast and accurate detection.
- **Remedy Suggestions**: Provides actionable advice for treating detected diseases.
- **PDF Reports**: (Simulated) Generate reports for your findings.

## ⚠️ Note
If you run the app *before* training the model, it will run in **Simulation Mode**, showing demo results to test the UI.
