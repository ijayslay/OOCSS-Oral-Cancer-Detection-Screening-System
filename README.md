# OOCSS — Oral Cancer Detection Screening System

OOCSS is an academic research prototype for exploring oral-cavity image classification and risk stratification. It combines a ResNet-50 image classifier with a Random Forest model that uses image probabilities and optional clinical factors.

> **Research use only.** OOCSS is not a medical device and does not diagnose cancer. Its outputs must not guide care or replace examination, diagnosis, or advice from a qualified clinician.

## What it does

- Classifies an uploaded image into **Normal**, **Benign**, **OPMD**, or **OSCC**.
- Displays the model's class probabilities.
- Optionally combines the image results with basic clinical factors for a triage-style risk estimate.
- Provides a local web interface and a command-line prediction script.

The labels mean:

| Label | Meaning |
| --- | --- |
| Normal | Healthy-appearing oral tissue |
| Benign | Non-malignant lesion |
| OPMD | Oral potentially malignant disorder |
| OSCC | Oral squamous cell carcinoma |

## Requirements

- Python **3.10 or 3.11** (recommended; the pinned PyTorch packages in `requirements.txt` target these versions)
- Git LFS to clone or push the ResNet-50 model file tracked in this repository
- Internet access for the first model startup if the standard ResNet-50 ImageNet checkpoint is not already cached

The trained project weights are in `models/`. On startup the code first constructs the ResNet-50 architecture using TorchVision's standard ImageNet checkpoint, then loads the OOCSS weights. TorchVision caches that checkpoint after its first download.

## Quick start: web app

Run these commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser. The API health endpoint is at `/health`, and the interactive API documentation is at `/docs`.

## Command-line prediction

```bash
python predict.py --image path/to/oral_photo.jpg
```

Optional clinical inputs can be supplied to the risk model:

```bash
python predict.py --image path/to/oral_photo.jpg --age 52 --sex 1 --tobacco 2 --alcohol 1 --betel 2 --duration 8 --size 12.5
```

The accepted values are defined by the argument help (`python predict.py --help`). Do not enter identifiable patient information into this research prototype.

## Model training

Pre-trained model files are included, so training is not required to launch the web app. To train from scratch or fine-tune, first provide a dataset and then run:

```bash
python test_setup.py
python train.py
```

Images should be arranged by split and class:

```text
data/
├── train/{Normal,Benign,OPMD,OSCC}/
├── val/{Normal,Benign,OPMD,OSCC}/
└── test/{Normal,Benign,OPMD,OSCC}/
```

Alternatively, configure Kaggle credentials and use `python download_dataset.py` as described in that script. Dataset files are intentionally excluded from Git because of their size and licensing requirements. Check the dataset's license and usage terms before downloading, sharing, or training on it.

Training writes model files to `models/` and plots and history files to `outputs/`. Training can take several hours on a CPU.

## Included model artifacts

- `models/resnet50_oocss.pth` — trained four-class ResNet-50 weights (tracked with Git LFS)
- `models/random_forest_oocss.joblib` — trained risk-stratification classifier

The Random Forest artifact was serialized with scikit-learn. Use the versions in `requirements.txt` when possible; loading serialized models with a different scikit-learn version can produce compatibility warnings or behavior changes.

## Project layout

```text
app.py                 FastAPI web application
predict.py             Single-image command-line prediction
train.py               Training entry point
download_dataset.py    Optional Kaggle dataset downloader
config.py              Model, data, and training configuration
models/                Trained model artifacts
outputs/               Training plots and history
static/                Web interface
utils/                 Dataset, model, training, evaluation, and risk code
```

## Limitations and responsible use

This project is for academic demonstration and research. Model performance can vary with image quality, population, equipment, and dataset composition. A probability or risk category from this software is not a diagnosis and should never be used to delay professional assessment. Any concerning oral lesion should be evaluated by a qualified healthcare professional.

Before redistributing this project, confirm that you have permission to share the model weights, dataset-derived outputs, and any included materials. No clinical or regulatory validation is implied.
