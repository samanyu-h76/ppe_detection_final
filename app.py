import os
import requests
import streamlit as st
import torch
import gdown
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

def download_weights_if_needed():
    # If a bad/corrupt file exists (often tiny HTML), delete and redownload
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) < 1_000_000:  # <1MB is definitely wrong for a .pt
            os.remove(MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights…")
        # You can provide either MODEL_URL or GDRIVE_FILE_ID in secrets
        if "MODEL_URL" in st.secrets:  # can be a normal Google Drive share link
            gdown.download(st.secrets["MODEL_URL"], MODEL_PATH, quiet=False, fuzzy=True)
        elif "GDRIVE_FILE_ID" in st.secrets:
            gdown.download(f"https://drive.google.com/uc?id={st.secrets['GDRIVE_FILE_ID']}",
                           MODEL_PATH, quiet=False)
        else:
            raise RuntimeError("Add MODEL_URL or GDRIVE_FILE_ID to Streamlit Secrets.")

download_weights_if_needed()

@st.cache_resource(show_spinner=True)
def load_model():
    # Pin YOLOv5 version and force hub reload to avoid stale cache
    mdl = torch.hub.load('ultralytics/yolov5:v6.2',
                         'custom',
                         path=MODEL_PATH,
                         force_reload=True)   # <- important
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
