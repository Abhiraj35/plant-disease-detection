# Plant Doctor AI 🌿

Plant Doctor AI is a plant disease detection system that classifies tomato leaf diseases in real-time. The frontend is built with **Streamlit**, while the machine learning model is trained using **Scikit-Learn** (comparing SVM, Random Forest, KNN, etc.) and deployed locally for high-performance inference.

## System Architecture
**User** → **Streamlit Frontend UI**
          ↓
  **Inference Engine (Local Python Process)**
  [Preprocessing (OpenCV) → Feature Extraction (Flatten) → Prediction (Scikit-Learn)]
          ↓
     **Result returned to UI**
          ↓
     **Remedy Suggestion (Mapped Results)**

## Features
- 🍃 **Tomato Leaf Disease Classification** — Detects diseases like Bacterial Spot, Early Blight, and Septoria Leaf Spot.
- ⚡ **Real-Time Predictions** — High-speed inference using trained Scikit-Learn models.
- 📸 **Dual Input Modes** — Supports image uploads and live camera capture.
- 💊 **Actionable Remedies** — Provides immediate treatment suggestions for detected diseases.
- 🧠 **Multi-Model Evaluation** — Compares performance of SVM, Random Forest, KNN, Naive Bayes, and Logistic Regression.

## Tech Stack
| Layer | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Language** | Python |
| **Machine Learning** | Scikit-Learn, OpenCV, PIL |
| **Model Format** | Joblib (.pkl) |
| **Data Processing** | NumPy, Pandas |

## Machine Learning Workflow
1. **Data Preprocessing** — Images resized to 64x64, color-converted (BGR to RGB), and flattened.
2. **Feature Extraction** — Raw pixel values used as features after flattening.
3. **Model Training** — Logistic Regression, SVM (RBF/Linear), Decision Tree, Random Forest, KNN, and Naive Bayes evaluated and compared.
4. **Evaluation** — Accuracy score on a 20% test split.
5. **Model Export** — Best performing model saved as `best_tomato_model.pkl`.
6. **Deployment** — Served via Streamlit for real-time inference.

## Getting Started

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

## Project Structure
```
Plant_Disease_Detection/
├── dataset/              # Training and testing images
├── app.py                # Main Streamlit UI
├── main.py               # Model training and comparison script
├── best_tomato_model.pkl # Saved Scikit-Learn model
├── tomato_class_names.json # Mapping of class indices to names
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```
## Deliverables
- [x] Structured Python Repository
- [x] Interactive Streamlit Application
- [x] Trained Scikit-Learn Model
- [x] Multi-Model Performance Report (Console Output)

## License
This project is open source.
