import streamlit as st
import os
import random
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

# --- 1. SETUP & MODEL LOADING ---
load_dotenv()
st.set_page_config(page_title="Pathology Validator", layout="wide")

MODEL_PATH = 'skin_disease_model.pth'
TEST_DIR = Path("data/test_data")

@st.cache_resource
def load_custom_model():
    """Loads the trained ResNet50 model and class names."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file {MODEL_PATH} not found. Please train the model first.")
        return None, None
    
    # Load checkpoint to CPU for accessibility
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    class_names = checkpoint['class_names']
    
    # Reconstruct architecture
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names

def predict_image(model, class_names, image_path):
    """Preprocesses image and returns prediction."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_t)
        _, index = torch.max(outputs, 1)
        
    return class_names[index[0]]

# Initialize Model
custom_model, classes = load_custom_model()

# --- 2. UI STYLING ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .header-text { color: #1a2a6c; border-bottom: 2px solid #1a2a6c; padding-bottom: 5px; margin-bottom: 15px; font-weight: bold; }
    div[data-testid="stMetric"] { background-color: #ffffff; border: 1px solid #d1d8e0; padding: 10px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("<h1 style='text-align: center; color: #1a2a6c;'>🩺 Clinical Performance Dashboard</h1>", unsafe_allow_html=True)

# --- 3. GLOBAL PERFORMANCE ---
if st.session_state.history:
    df_total = pd.DataFrame(st.session_state.history)
    correct_preds = len(df_total[df_total['Actual'] == df_total['Predicted']])
    total_samples = len(df_total)
    global_accuracy = (correct_preds / total_samples) * 100 if total_samples > 0 else 0
    
    g1, g2 = st.columns(2)
    with g1: st.metric("📋 Total Samples Tested", total_samples)
    with g2: st.metric("📈 Global Accuracy", f"{global_accuracy:.1f}%")
    st.markdown("---")

# --- 4. ANALYSIS INTERFACE ---
if classes:

    # Filter classes to only those present in test directory
    available_classes = [c for c in classes if (TEST_DIR / c).exists()]
    
    selected_class = None
    if available_classes:
        selected_class = st.selectbox("🎯 Target Category to Test:", classes)
        class_path = TEST_DIR / selected_class

    else:
        st.error("No test data found for any model classes.")
        class_path = Path("non_existent_path")
        
    if class_path.exists():
        images = [f.name for f in class_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        if images:
            if 'current_img' not in st.session_state or st.button("🔄 Shuffle New Sample"):
                st.session_state.current_img = random.choice(images)
                st.session_state.last_prediction = None

            img_path = (class_path / st.session_state.current_img).resolve()
            col_img, col_info = st.columns([1, 1.5]) 
            
            with col_img:
                st.markdown("<p class='header-text'>Specimen View</p>", unsafe_allow_html=True)
                st.image(Image.open(img_path), use_container_width=True)
            
            with col_info:
                st.markdown("<p class='header-text'>AI Diagnostic</p>", unsafe_allow_html=True)
                st.info(f"**Ground Truth Label:** {selected_class}")
                
                if st.button("🚀 Run AI Analysis", use_container_width=True):
                    prediction = predict_image(custom_model, classes, img_path)
                    st.session_state.last_prediction = prediction
                    # Save to history
                    st.session_state.history.append({"Actual": selected_class, "Predicted": prediction})
                    st.rerun()

                if st.session_state.get('last_prediction'):
                    if st.session_state.last_prediction == selected_class:
                        st.success(f"**AI PREDICTION:** {st.session_state.last_prediction} ✅")
                    else:
                        st.error(f"**AI PREDICTION:** {st.session_state.last_prediction} ❌")

    # --- 5. PER-CLASS RELIABILITY (TP, TN, FP, FN) ---
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # Binary Classification Logic for the Selected Class:
        # TP: Actual is target, Predicted is target
        tp = len(df[(df['Actual'] == selected_class) & (df['Predicted'] == selected_class)])
        # FN: Actual is target, Predicted is NOT target
        fn = len(df[(df['Actual'] == selected_class) & (df['Predicted'] != selected_class)])
        # FP: Actual is NOT target, Predicted is target
        fp = len(df[(df['Actual'] != selected_class) & (df['Predicted'] == selected_class)])
        # TN: Actual is NOT target, Predicted is NOT target
        tn = len(df[(df['Actual'] != selected_class) & (df['Predicted'] != selected_class)])

        st.markdown(f"<p class='header-text'>Category Stats: {selected_class.upper()}</p>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✅ TP (Correct Hits)", tp)
        m2.metric("🛡️ TN (Correct Rejections)", tn)
        m3.metric("⚠️ FP (False Alarms)", fp)
        m4.metric("❌ FN (Misses)", fn)

# --- 6. LOGS & ADMIN ---
if st.session_state.history:
    with st.expander("📜 Historical Audit Log"):
        st.dataframe(pd.DataFrame(st.session_state.history).tail(10), use_container_width=True)

st.sidebar.markdown("### ⚙️ System Settings")
if st.sidebar.button("🗑️ Reset All Metrics"):
    st.session_state.history = []
    st.rerun()