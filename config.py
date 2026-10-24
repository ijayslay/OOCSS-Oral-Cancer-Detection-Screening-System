"""
config.py
=========
Single source of truth for every setting in the OOCSS project.
Change values HERE — never scatter magic numbers through other files.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "val"
TEST_DIR    = DATA_DIR / "test"
MODEL_DIR   = BASE_DIR / "models"
OUTPUT_DIR  = BASE_DIR / "outputs"

for d in [MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Classes ──────────────────────────────────────────────────────────────────
# ORDER MATTERS — index 0 = Normal, 1 = Benign, 2 = OPMD, 3 = OSCC
CLASS_NAMES = ["Normal", "Benign", "OPMD", "OSCC"]
NUM_CLASSES = len(CLASS_NAMES)

# ── Image settings ───────────────────────────────────────────────────────────
IMG_SIZE    = 224          # ResNet-50 expects 224×224 pixels
CHANNELS    = 3            # RGB

# ImageNet mean and std — used for normalisation because we use
# a ResNet pre-trained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE      = 16       # Smaller batch = less RAM — safe for CPU
NUM_EPOCHS      = 30       # Reduced for CPU; increase on GPU
LEARNING_RATE   = 1e-4     # Adam initial learning rate
LR_PATIENCE     = 5        # Reduce LR after this many epochs with no improvement
EARLY_STOP_PAT  = 10       # Stop training after this many epochs with no improvement
DROPOUT_RATE    = 0.4      # Regularisation in classification head
SEED            = 42       # Reproducibility

# ── ResNet-50 fine-tuning ─────────────────────────────────────────────────────
# How many layers to FREEZE (not train) from the beginning of the network.
# ResNet-50 has ~177 named children parameters.
# Freezing the first 60 keeps stem + layer1 + layer2 frozen.
FREEZE_LAYERS   = 60

# ── Random Forest hyperparameters ────────────────────────────────────────────
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH    = 12
RF_MIN_SAMPLES  = 10
RF_RANDOM_STATE = 42

# ── Risk threshold ────────────────────────────────────────────────────────────
# RF output score thresholds for triage category
# Score < LOW_THRESH  → Low risk
# LOW_THRESH ≤ score < HIGH_THRESH → Moderate risk
# score ≥ HIGH_THRESH → High risk (refer immediately)
LOW_THRESH  = 0.35
HIGH_THRESH = 0.65

# ── Device ────────────────────────────────────────────────────────────────────
# "cpu" — you are on a CPU-only machine.
# Change to "cuda" if you ever move to a GPU machine.
DEVICE = "cpu"

# ── Augmentation (training only) ──────────────────────────────────────────────
AUG_ROTATION    = 15       # degrees
AUG_BRIGHTNESS  = 0.2
AUG_CONTRAST    = 0.2
AUG_HFLIP_PROB  = 0.5

# ── Saved file names ──────────────────────────────────────────────────────────
CNN_WEIGHTS_FILE   = MODEL_DIR / "resnet50_oocss.pth"
RF_MODEL_FILE      = MODEL_DIR / "random_forest_oocss.joblib"
HISTORY_FILE       = OUTPUT_DIR / "training_history.csv"
