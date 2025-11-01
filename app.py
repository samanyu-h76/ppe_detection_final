import streamlit as st
import torch
from PIL import Image
import os

# Load YOLOv5 model
MODEL_PATH = 'yolov5/runs/train/PPE-Detection3/weights/best.pt'  # change if your best.pt is in another folder
model = torch.hub.load('yolov5', 'custom', path=MODEL_PATH, source='local')

# Streamlit UI
st.set_page_config(page_title="PPE Detection System", page_icon="🦺", layout="wide")
st.title("🦺 PPE Compliance Detection System")
st.write("Upload an image to detect PPE items like helmets, masks, and vests.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# --- Compliance Logic ---
def get_compliance(detections):
    """
    detections: list of detected class names
    """
    # Presence flags
    has_hat = 'Hardhat' in detections
    has_mask = 'Mask' in detections
    has_vest = 'Safety Vest' in detections

    # Absence flags
    no_hat = 'NO-Hardhat' in detections
    no_mask = 'NO-Mask' in detections
    no_vest = 'NO-Safety Vest' in detections

    # Compliance decision
    if has_hat and has_mask and has_vest and not (no_hat or no_mask or no_vest):
        return "🟢 Fully Compliant"
    elif any([no_hat, no_mask, no_vest]) and not (no_hat and no_mask and no_vest):
        return "🟡 Partially Compliant"
    elif no_hat and no_mask and no_vest:
        return "🔴 Non-Compliant"
    else:
        return "🟡 Partially Compliant"  # fallback

# --- Streamlit Interface ---
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO inference
    results = model(image)
    detections = results.pandas().xyxy[0]['name'].tolist()

    st.subheader("Detection Results")
    st.image(results.render()[0], caption="Detected PPE Items", use_column_width=True)

    st.write("**Detected Objects:**", ", ".join(set(detections)))

    compliance_status = get_compliance(detections)
    st.markdown(f"### Compliance Status: {compliance_status}")

    # Save detections
    os.makedirs("detections", exist_ok=True)
    results.save(save_dir="detections")