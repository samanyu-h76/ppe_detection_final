# ---- stdlib shadowing + bad backport guard (top of app.py) ----
import sys, importlib, subprocess

# Ensure stdlib/site-packages come before CWD so local files don't shadow stdlib
if sys.path:
    cwd = sys.path[0]
    sys.path = sys.path[1:] + [cwd]

# If a third-party 'pathlib' package is installed (a directory with __init__.py),
# uninstall it so the real stdlib 'pathlib.py' is used.
try:
    import pathlib as _pl
    pl_file = getattr(_pl, "__file__", "") or ""
    if "site-packages" in pl_file and pl_file.endswith("pathlib/__init__.py"):
        # remove the backport package that breaks YOLOv5 imports
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pathlib"], check=False)
        importlib.invalidate_caches()
        del sys.modules["pathlib"]
        import pathlib as _pl  # reload stdlib module
except Exception:
    pass
# ---------------------------------------------------------------

import os
from pathlib import Path
import subprocess
import streamlit as st
import gdown
import torch
from PIL import Image

# ---------- download weights once ----------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
os.makedirs(MODEL_DIR, exist_ok=True)

def download_weights_if_needed():
    # delete corrupt HTML downloads
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 1_000_000:
        os.remove(MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights…")
        if "MODEL_URL" in st.secrets:
            # gdown handles normal Drive share links
            gdown.download(st.secrets["MODEL_URL"], MODEL_PATH, quiet=False, fuzzy=True)
        elif "GDRIVE_FILE_ID" in st.secrets:
            gdown.download(f"https://drive.google.com/uc?id={st.secrets['GDRIVE_FILE_ID']}",
                           MODEL_PATH, quiet=False)
        else:
            st.error("Add MODEL_URL or GDRIVE_FILE_ID to Streamlit Secrets.")
            st.stop()

download_weights_if_needed()

# ---------- load YOLOv5 model safely ----------
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        mdl = torch.hub.load(
            'ultralytics/yolov5:v6.2',
            'custom',
            path=MODEL_PATH,
            force_reload=True,
            trust_repo=True,     # avoid GitHub API validation errors
        )
    except Exception as e:
        # fallback: shallow clone v6.2 locally then load as local
        st.warning(f"Torch Hub failed ({e}). Cloning YOLOv5 v6.2 locally…")
        repo_dir = Path("yolov5_v62_local")
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", "v6.2",
                 "https://github.com/ultralytics/yolov5.git", str(repo_dir)],
                check=True,
            )
        mdl = torch.hub.load(
            str(repo_dir),
            'custom',
            path=MODEL_PATH,
            source='local',
            force_reload=True,
        )
    # make detections a bit easier to see
    mdl.conf = 0.20   # lower if you still see no boxes (e.g., 0.15)
    mdl.iou = 0.45
    return mdl

try:
    model = load_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()   # <- prevents NameError later

# ---------- UI ----------
st.set_page_config(page_title="PPE Detection System", page_icon="🦺", layout="wide")
st.title("🦺 PPE Compliance Detection System")
st.write("Upload an image to detect PPE items like helmets, masks, and vests.")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# ---------- compliance logic ----------
def get_compliance(detections):
    has_hat  = 'Hardhat' in detections
    has_mask = 'Mask' in detections
    has_vest = 'Safety Vest' in detections
    no_hat  = 'NO-Hardhat' in detections
    no_mask = 'NO-Mask' in detections
    no_vest = 'NO-Safety Vest' in detections

    if has_hat and has_mask and has_vest and not (no_hat or no_mask or no_vest):
        return "🟢 Fully Compliant"
    elif no_hat and no_mask and no_vest:
        return "🔴 Non-Compliant"
    else:
        return "🟡 Partially Compliant"

# ---------- inference ----------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # run YOLO and render boxes
    results = model(image, size=640)
    df = results.pandas().xyxy[0]
    names = df['name'].tolist()

    rendered = results.render()[0]          # numpy array (RGB)
    st.subheader("Detection Results")
    st.image(rendered, caption="Detected PPE Items", use_column_width=True)

    st.write("**Detected Objects:**", ", ".join(sorted(set(names))) or "None")
    st.markdown(f"### Compliance Status: {get_compliance(names)}")

    os.makedirs("detections", exist_ok=True)
    results.save(save_dir="detections")
