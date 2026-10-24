"""
predict.py
==========
Run OOCSS inference on a single oral cavity image.
This is what a clinician or health worker would use in the field.

Usage (in VS Code terminal):
  python predict.py --image path/to/image.jpg

Optional clinical covariates (improve the RF risk score):
  python predict.py --image photo.jpg \\
      --age 52 --sex 1 --tobacco 2 --alcohol 1 --betel 2 \\
      --duration 8 --size 12.5

Arguments:
  --image      Path to an oral cavity image (JPG or PNG)
  --age        Patient age in years          (default: 45)
  --sex        0 = female, 1 = male          (default: 0)
  --tobacco    0=never 1=former 2=smoke 3=smokeless  (default: 0)
  --alcohol    0=never 1=occasional 2=heavy  (default: 0)
  --betel      0=never 1=occasional 2=regular (default: 0)
  --duration   Lesion duration in weeks      (default: 4)
  --size       Lesion size in mm             (default: 5.0)
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils.model      import load_model
from utils.risk_model import build_feature_vector, load_rf, score_to_triage


# ── Image pre-processing (must match training eval_transform) ─────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
])


# ── Load image ─────────────────────────────────────────────────────────────────
def load_image(image_path: str) -> torch.Tensor:
    img  = Image.open(image_path).convert("RGB")
    tensor = INFER_TRANSFORM(img)
    return tensor.unsqueeze(0)   # add batch dimension: [1, 3, 224, 224]


# ── CNN inference ──────────────────────────────────────────────────────────────
@torch.no_grad()
def cnn_predict(model, image_tensor: torch.Tensor):
    model.eval()
    image_tensor = image_tensor.to(config.DEVICE)
    logits  = model(image_tensor)
    probs   = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(probs.argmax())
    return probs, pred_idx


# ── Full pipeline ──────────────────────────────────────────────────────────────
def run_inference(args):
    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"[ERROR] Image not found: {image_path}")

    # ── Check models exist ────────────────────────────────────────────────
    if not config.CNN_WEIGHTS_FILE.exists():
        sys.exit(
            f"[ERROR] CNN weights not found at {config.CNN_WEIGHTS_FILE}.\n"
            "        Please run: python train.py"
        )
    if not config.RF_MODEL_FILE.exists():
        sys.exit(
            f"[ERROR] RF model not found at {config.RF_MODEL_FILE}.\n"
            "        Please run: python train.py"
        )

    print(f"\n╔══════════════════════════════════════════════════╗")
    print(f"║          OOCSS — Oral Cancer Screening           ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print(f"\n  Image: {image_path.name}")
    print(f"  Patient age: {args.age}  |  Sex: {'M' if args.sex else 'F'}")

    # ── Step 1: Load models ───────────────────────────────────────────────
    print("\n[→] Loading models …")
    model = load_model(config.CNN_WEIGHTS_FILE)
    rf    = load_rf(config.RF_MODEL_FILE)

    # ── Step 2: CNN prediction ────────────────────────────────────────────
    print("[→] Running image classification …")
    image_tensor        = load_image(str(image_path))
    probs, pred_idx     = cnn_predict(model, image_tensor)
    predicted_class     = config.CLASS_NAMES[pred_idx]
    confidence          = probs[pred_idx] * 100

    print("\n── Image Classification Result ───────────────────────")
    for i, (cls, p) in enumerate(zip(config.CLASS_NAMES, probs)):
        marker = "◄" if i == pred_idx else " "
        bar    = "█" * int(p * 30)
        print(f"  {cls:<12}  {p*100:>6.2f}%  {bar} {marker}")

    print(f"\n  Predicted: {predicted_class}  (confidence: {confidence:.1f}%)")

    # ── Step 3: Risk stratification ───────────────────────────────────────
    print("\n[→] Running clinical risk stratification …")
    feature_vector = build_feature_vector(
        image_probs    = probs,
        age            = args.age,
        sex            = args.sex,
        tobacco_use    = args.tobacco,
        alcohol_use    = args.alcohol,
        betel_quid     = args.betel,
        lesion_duration= args.duration,
        lesion_size_mm = args.size,
    )
    rf_probs   = rf.predict_proba(feature_vector.reshape(1, -1))[0]
    oscc_score = float(rf_probs[3])    # probability of OSCC class
    triage     = score_to_triage(oscc_score)

    print("\n── Risk Stratification Result ────────────────────────")
    print(f"  OSCC risk score : {oscc_score:.3f}")
    print(f"  Triage decision : {triage}")

    # ── Final recommendation ───────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════")
    print("  CLINICAL RECOMMENDATION")
    print("══════════════════════════════════════════════════════")

    if pred_idx == 0 and oscc_score < config.LOW_THRESH:
        print("  ✅  No suspicious lesion detected.")
        print("      Routine recall in 6–12 months.")

    elif pred_idx == 1 or (pred_idx == 0 and oscc_score < config.HIGH_THRESH):
        print("  ⚠️   Benign lesion or elevated baseline risk.")
        print("      Clinical review recommended within 4 weeks.")
        print("      Document and monitor lesion progression.")

    elif pred_idx == 2 or oscc_score >= config.LOW_THRESH:
        print("  🔴  Potentially malignant disorder (OPMD) detected.")
        print("      PRIORITY referral to oral medicine specialist.")
        print("      Biopsy required for definitive diagnosis.")

    else:
        print("  🚨  HIGH SUSPICION FOR OSCC.")
        print("      URGENT referral — specialist within 2 days.")
        print("      Do NOT wait for biopsy to initiate referral.")

    print("\n  ⚠️  DISCLAIMER: This is a screening tool only.")
    print("     All findings must be confirmed by a qualified clinician.")
    print("══════════════════════════════════════════════════════\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="OOCSS — Oral Cancer Detection Screening Tool"
    )
    p.add_argument("--image",    required=True,  type=str)
    p.add_argument("--age",      default=45,     type=int)
    p.add_argument("--sex",      default=0,      type=int,   choices=[0, 1])
    p.add_argument("--tobacco",  default=0,      type=int,   choices=[0, 1, 2, 3])
    p.add_argument("--alcohol",  default=0,      type=int,   choices=[0, 1, 2])
    p.add_argument("--betel",    default=0,      type=int,   choices=[0, 1, 2])
    p.add_argument("--duration", default=4,      type=int)
    p.add_argument("--size",     default=5.0,    type=float)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
