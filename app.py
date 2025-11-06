import os
import requests
import streamlit as st
import torch
from PIL import Image

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="PPE Detection System", page_icon="🦺", layout="wide")
st.title("🦺 PPE Compliance Detection System")
st.write("Upload an image to detect PPE items like helmets, masks, and vests.")

# -------------------------------
# Download model from URL in secrets (only once)
# -------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    # Put a direct-download link in Streamlit Cloud → Settings → Secrets: MODEL_URL="https://drive.google.com/uc?export=download&id=FILEID"
    url = st.secrets["MODEL_URL"]
    st.info("Downloading model weights… first run may take a moment.")
    r = requests.get(url, allow_redirects=True)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# -------------------------------
# Load YOLOv5 model (from GitHub, not local folder)
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Pin to YOLOv5 v6.2 which doesn't require the ultralytics package
    mdl = torch.hub.load('ultralytics/yolov5:v6.2', 'custom', path=MODEL_PATH)
    mdl.conf = 0.25
    return mdl


model = load_model()

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# -------------------------------
# Compliance Logic
# -------------------------------
def get_compliance(detections):
    """
    detections: list of detected class names (strings)
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
    elif no_hat and no_mask and no_vest:
        return "🔴 Non-Compliant"
    else:
        return "🟡 Partially Compliant"  # one or two missing, or mixed signals

# -------------------------------
# Inference
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)
    detections = results.pandas().xyxy[0]['name'].tolist()

    st.subheader("Detection Results")
    st.image(results.render()[0], caption="Detected PPE Items", use_column_width=True)

    st.write("**Detected Objects:**", ", ".join(sorted(set(detections))) or "None")
    st.markdown(f"### Compliance Status: {get_compliance(detections)}")

    # Save detections (optional)
    os.makedirs("detections", exist_ok=True)
    results.save(save_dir="detections")
