"""
utils/risk_model.py
===================
Stage 2 of OOCSS: the Random Forest risk-stratification module.

What is a Random Forest?
------------------------
A Random Forest is an ensemble of many decision trees.
Each tree is trained on a random subset of the training data
(bootstrap sampling) and a random subset of features.
The final prediction is the majority vote across all trees.

Random Forests are great for:
  - Tabular data (numbers, categories)
  - Handling a mix of numerical and categorical features
  - Producing calibrated probability scores
  - Being interpretable via feature importance

In OOCSS, the RF takes:
  - The 4 class probabilities from ResNet-50 (image-derived)
  - 7 clinical covariates (age, sex, risk factors)
and produces a final risk score used for triage decisions.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib                              # for saving/loading sklearn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Feature names (must match input order in build_feature_vector) ────────────
IMAGE_PROB_FEATURES = [
    "prob_Normal", "prob_Benign", "prob_OPMD", "prob_OSCC"
]
CLINICAL_FEATURES = [
    "age",              # integer, years
    "sex",              # 0 = female, 1 = male
    "tobacco_use",      # 0=never, 1=former, 2=current_smoke, 3=smokeless
    "alcohol_use",      # 0=never, 1=occasional, 2=heavy
    "betel_quid",       # 0=never, 1=occasional, 2=regular
    "lesion_duration",  # integer, weeks
    "lesion_size_mm",   # float, millimetres
]
ALL_FEATURES = IMAGE_PROB_FEATURES + CLINICAL_FEATURES


# ── Build feature vector from a single patient ─────────────────────────────────

def build_feature_vector(
    image_probs   : np.ndarray,     # shape [4] — ResNet softmax output
    age           : int,
    sex           : int,            # 0=female, 1=male
    tobacco_use   : int,            # 0–3
    alcohol_use   : int,            # 0–2
    betel_quid    : int,            # 0–2
    lesion_duration: int,           # weeks
    lesion_size_mm : float,         # mm
) -> np.ndarray:
    """
    Concatenates image probabilities and clinical covariates into a
    single 1-D feature vector of length 11.
    """
    clinical = np.array([
        age, sex, tobacco_use, alcohol_use,
        betel_quid, lesion_duration, lesion_size_mm,
    ], dtype=np.float32)
    return np.concatenate([image_probs.astype(np.float32), clinical])


# ── Generate synthetic training data (used when no real CSV exists) ────────────

def generate_synthetic_rf_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Creates a realistic synthetic dataset for training the RF module.

    In a real project you would replace this with actual patient records.
    This function lets you run and test the pipeline end-to-end immediately.

    The synthetic data encodes known clinical relationships:
      - Higher tobacco/betel use → higher OSCC probability
      - Older patients tend toward OSCC
      - OPMD patients have intermediate-risk profiles
    """
    rng = np.random.default_rng(seed)
    rows = []

    # We generate samples for each label (0=Normal, 1=Benign, 2=OPMD, 3=OSCC)
    samples_per_class = n_samples // 4
    label_configs = {
        # label: (prob_mean[4], age_range, tobacco_weight, alcohol_weight, betel_weight)
        0: ([0.75, 0.15, 0.07, 0.03], (20, 55), [0.70, 0.20, 0.07, 0.03], [0.70, 0.20, 0.10], [0.80, 0.15, 0.05]),
        1: ([0.15, 0.65, 0.15, 0.05], (25, 60), [0.55, 0.25, 0.12, 0.08], [0.60, 0.25, 0.15], [0.70, 0.20, 0.10]),
        2: ([0.08, 0.15, 0.60, 0.17], (35, 70), [0.30, 0.20, 0.25, 0.25], [0.45, 0.30, 0.25], [0.40, 0.35, 0.25]),
        3: ([0.04, 0.08, 0.18, 0.70], (45, 80), [0.15, 0.15, 0.30, 0.40], [0.30, 0.30, 0.40], [0.25, 0.35, 0.40]),
    }

    for label, (prob_mean, age_range, tob_w, alc_w, bet_w) in label_configs.items():
        for _ in range(samples_per_class):
            # Simulate softmax probabilities with Dirichlet noise
            probs = rng.dirichlet(np.array(prob_mean) * 10)

            age             = int(rng.integers(*age_range))
            sex             = int(rng.integers(0, 2))
            tobacco_use     = int(rng.choice(4, p=tob_w))
            alcohol_use     = int(rng.choice(3, p=alc_w))
            betel_quid      = int(rng.choice(3, p=bet_w))
            lesion_duration = int(rng.integers(1, 52 if label < 2 else 104))
            lesion_size_mm  = float(rng.uniform(2, 8 if label < 2 else 25))

            rows.append({
                **dict(zip(IMAGE_PROB_FEATURES, probs)),
                "age"             : age,
                "sex"             : sex,
                "tobacco_use"     : tobacco_use,
                "alcohol_use"     : alcohol_use,
                "betel_quid"      : betel_quid,
                "lesion_duration" : lesion_duration,
                "lesion_size_mm"  : lesion_size_mm,
                "label"           : label,
            })

    df = pd.DataFrame(rows)
    # Shuffle
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Train the Random Forest ────────────────────────────────────────────────────

def train_rf(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Trains the Random Forest on a DataFrame with columns matching
    ALL_FEATURES + 'label'.

    Returns the trained classifier.
    """
    X = df[ALL_FEATURES].values
    y = df["label"].values

    # 80/20 split for quick validation
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RF_RANDOM_STATE
    )

    print("[→] Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators = config.RF_N_ESTIMATORS,
        max_depth    = config.RF_MAX_DEPTH,
        min_samples_leaf = config.RF_MIN_SAMPLES,
        class_weight = "balanced",      # auto-weights rare classes
        random_state = config.RF_RANDOM_STATE,
        n_jobs       = -1,              # use all CPU cores
    )
    rf.fit(X_tr, y_tr)

    # ── Evaluation ─────────────────────────────────────────────────────────
    y_pred  = rf.predict(X_val)
    y_proba = rf.predict_proba(X_val)
    acc     = accuracy_score(y_val, y_pred) * 100

    # AUC: one-vs-rest for multiclass
    auc = roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro")

    print(f"[✓] RF trained.  Val accuracy: {acc:.1f}%  |  Macro AUC: {auc:.3f}")
    print("\n" + classification_report(
        y_val, y_pred,
        target_names=config.CLASS_NAMES,
    ))

    return rf


# ── Feature importance ─────────────────────────────────────────────────────────

def print_feature_importance(rf: RandomForestClassifier):
    """Prints a ranked table of feature importances."""
    importances = rf.feature_importances_
    pairs = sorted(zip(ALL_FEATURES, importances), key=lambda x: x[1], reverse=True)
    print("\n── Random Forest Feature Importances ────────────────")
    for name, imp in pairs:
        bar = "█" * int(imp * 50)
        print(f"  {name:<20} {imp:.4f}  {bar}")
    print()


# ── Risk score → triage label ──────────────────────────────────────────────────

def score_to_triage(score: float) -> str:
    """
    Converts the RF's OSCC probability score to a triage recommendation.
    'score' is the predicted probability for class OSCC (index 3).
    """
    if score < config.LOW_THRESH:
        return "LOW — routine follow-up"
    elif score < config.HIGH_THRESH:
        return "MODERATE — priority dental review within 2 weeks"
    else:
        return "HIGH — urgent specialist referral"


# ── Save / load ────────────────────────────────────────────────────────────────

def save_rf(rf: RandomForestClassifier, path: Path = config.RF_MODEL_FILE):
    joblib.dump(rf, path)
    print(f"[✓] RF model saved → {path}")


def load_rf(path: Path = config.RF_MODEL_FILE) -> RandomForestClassifier:
    rf = joblib.load(path)
    print(f"[✓] RF model loaded ← {path}")
    return rf
