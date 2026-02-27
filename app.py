import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import time
import joblib
import cv2

st.set_page_config(
    page_title="Plant Disease Detection AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .report-box {
        padding: 20px;
        color: #4CAF50;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .highlight {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.title("Plant Doctor AI 🌿")
    st.info("Advanced ML Comparison System (SVM, RF, KNN).")
    
    st.markdown("---")
    mode = st.radio("Select Input Mode:", ["Upload Image", "Use Camera"])
    st.markdown("---")
    st.caption("Powered by Scikit-Learn")

st.title("🍃 Plant Disease Detection System")
st.markdown("### Detect diseases. Get remedies. Save your crop.")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_tomato_model.pkl')
        return model
    except Exception as e:
        return None

model = load_model()

import json

def load_class_names():
    try:
        with open('tomato_class_names.json', 'r') as f:
            return json.load(f)
    except:
        return []

CLASS_NAMES = load_class_names()

if model is None:
    st.warning("⚠️ Model not found! Please run `train_model.ipynb` to generate `best_model.pkl` first.")
    st.info("For demonstration, the app will simulate predictions if no model is loaded.")
elif not CLASS_NAMES:
    st.warning("⚠️ Class names not found! Please run `train_model.ipynb` to generate `class_names.json`.")

REMEDIES = {
    "Apple_scab": "Apply fungicides like captan or myclobutanil. Prune and destroy infected leaves.",
    "Black_rot": "Remove mummified fruit. Use fungicides. Maintain good air circulation.",
    "Bacterial_spot": "Use copper-based bactericides. Remove infected plant parts immediately.",
    "Early_blight": "Apply copper fungicides. Rotate crops. Remove lower infected leaves.",
    "Late_blight": "Destroy infected plants. Use resistant varieties. Apply fungicides preventively.",
    "Powdery_mildew": "Use sulfur or neem oil. Improving air circulation helps reduce spread.",
    "healthy": "Your plant looks healthy! Keep up the good work with regular watering and sunlight."
}

def get_remedy(disease_name):
    for key, remedy in REMEDIES.items():
        if key in disease_name:
            return remedy
    return "Consult a local agriculturist for specific treatment."

input_image = None
IMG_SIZE = 64 # Must match training size

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
elif mode == "Use Camera":
    camera_input = st.camera_input("Take a picture")
    if camera_input is not None:
        input_image = Image.open(camera_input)

if input_image:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Your Image")
        st.image(input_image, use_column_width=True, caption="Analysis Target")
    
    with col2:
        st.subheader("Analysis Results")
        
        with st.spinner("Processing & Analyzing..."):
            time.sleep(1) 
            
            img_array = np.array(input_image.convert('RGB')) 
            img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            img_flat = img_resized.flatten().reshape(1, -1)

            confidence = 0.0
            predicted_class = "Unknown"
            probs = None

            if model and CLASS_NAMES:
                try:
                    prediction_index = model.predict(img_flat)[0]
                    predicted_class = CLASS_NAMES[prediction_index]
                    
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(img_flat)[0]
                        confidence = np.max(probs)
                    else:
                        confidence = 1.0
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
            else:
                st.warning("Running in simulation mode (No model loaded)")
                predicted_class = "Tomato___Early_blight" 
                confidence = 0.88
                if CLASS_NAMES:
                   probs = np.zeros(len(CLASS_NAMES))
                   probs[0] = 0.88 

            st.markdown(f"""
            <div class='report-box'>
                <h3>Status: <span class='highlight'>{predicted_class.replace("___", " - ").replace("_", " ")}</span></h3>
                <h4>Confidence: {confidence*100:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)

            if probs is not None and len(CLASS_NAMES) == len(probs):
                st.caption("Confidence Distribution")
                chart_data = pd.DataFrame(
                    probs,
                    index=[c.split('___')[-1] for c in CLASS_NAMES],
                    columns=["Probability"]
                )
                top_5 = chart_data.sort_values(by="Probability", ascending=False).head(5)
                st.bar_chart(top_5)

    st.markdown("---")
    st.subheader("Diagnosis & Recommended Treatment")
    
    remedy = get_remedy(predicted_class)
    
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.markdown(f"""
        <div style='background-color: #e8f5e9;color: #4CAF50; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
            <h4>Expert Suggestion</h4>
            <p style='font-size: 16px;'>{remedy}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_r2:
        st.button("📄 Download PDF Report")
        st.caption("Generate a detailed diagnosis report for your records.")
