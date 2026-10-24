"""
app.py  —  OOCSS FastAPI Backend
=================================
Run with:
  poetry run uvicorn app:app --reload --port 8000
"""

import io
import sys
import base64
import traceback
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils.model      import build_resnet50
from utils.risk_model import build_feature_vector, load_rf, score_to_triage

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="OOCSS — Oral Cancer Screening API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Load models at startup ─────────────────────────────────────────────────────
cnn_model = None
rf_model  = None

@app.on_event("startup")
async def load_models():
    global cnn_model, rf_model
    print("[→] Loading CNN model …")
    cnn_model = build_resnet50()
    if config.CNN_WEIGHTS_FILE.exists():
        cnn_model.load_state_dict(
            torch.load(config.CNN_WEIGHTS_FILE, map_location=config.DEVICE)
        )
        cnn_model.eval()
        print("[✓] CNN model loaded.")
    else:
        print("[!] WARNING: CNN weights not found. Run train.py first.")

    print("[→] Loading Random Forest …")
    if config.RF_MODEL_FILE.exists():
        rf_model = load_rf(config.RF_MODEL_FILE)
        print("[✓] RF model loaded.")
    else:
        print("[!] WARNING: RF model not found. Run train.py first.")

# ── Image transform ────────────────────────────────────────────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
])

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cnn_loaded": cnn_model is not None,
        "rf_loaded" : rf_model  is not None,
    }

@app.post("/predict")
async def predict(
    file          : UploadFile = File(...),
    age           : int   = Form(45),
    sex           : int   = Form(0),
    tobacco_use   : int   = Form(0),
    alcohol_use   : int   = Form(0),
    betel_quid    : int   = Form(0),
    lesion_duration: int  = Form(4),
    lesion_size_mm : float = Form(5.0),
):
    if cnn_model is None or rf_model is None:
        raise HTTPException(503, "Models not loaded. Run train.py first.")

    # ── Read & validate image ──────────────────────────────────────────────
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    # ── CNN inference ──────────────────────────────────────────────────────
    try:
        tensor  = INFER_TRANSFORM(image).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            logits = cnn_model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx   = int(probs.argmax())
        pred_class = config.CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx]) * 100
    except Exception as e:
        raise HTTPException(500, f"CNN inference failed: {e}")

    # ── RF risk stratification ─────────────────────────────────────────────
    try:
        feat_vec   = build_feature_vector(
            probs, age, sex, tobacco_use, alcohol_use,
            betel_quid, lesion_duration, lesion_size_mm
        )
        rf_probs   = rf_model.predict_proba(feat_vec.reshape(1, -1))[0]
        oscc_score = float(rf_probs[3])
        triage     = score_to_triage(oscc_score)
    except Exception as e:
        raise HTTPException(500, f"RF inference failed: {e}")

    # ── Determine recommendation ───────────────────────────────────────────
    if pred_idx == 0 and oscc_score < config.LOW_THRESH:
        risk_level      = "low"
        recommendation  = "No suspicious lesion detected. Routine recall in 6–12 months."
    elif pred_idx == 1 or (pred_idx == 0 and oscc_score < config.HIGH_THRESH):
        risk_level      = "moderate"
        recommendation  = "Benign lesion detected. Clinical review recommended within 4 weeks. Monitor for progression."
    elif pred_idx == 2 or oscc_score >= config.LOW_THRESH:
        risk_level      = "high"
        recommendation  = "Potentially malignant disorder (OPMD) detected. Priority referral to oral medicine specialist. Biopsy required."
    else:
        risk_level      = "critical"
        recommendation  = "High suspicion for OSCC. Urgent specialist referral required within 48 hours."

    # ── Return structured response ─────────────────────────────────────────
    return JSONResponse({
        "predicted_class" : pred_class,
        "confidence"      : round(confidence, 2),
        "risk_level"      : risk_level,
        "oscc_score"      : round(oscc_score * 100, 1),
        "triage"          : triage,
        "recommendation"  : recommendation,
        "probabilities"   : {
            config.CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(config.CLASS_NAMES))
        },
        "disclaimer": "This is a screening tool only. All findings must be confirmed by a qualified clinician.",
    })
