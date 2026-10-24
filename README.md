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
- Internet access for training if the standard ResNet-50 ImageNet checkpoint is not already cached. Running the included trained models does not need that extra download.

The trained project weights are in `models/`. The web app loads the saved OOCSS weights directly. Training still initializes ResNet-50 with TorchVision's ImageNet checkpoint.

## Get the code and model

Install [Git LFS](https://git-lfs.com/) before cloning so Git downloads the ResNet-50 model file instead of leaving an LFS pointer in its place. Then run:

```bash
git lfs install
git clone https://github.com/ijayslay/OOCSS-Oral-Cancer-Detection-Screening-System.git
cd OOCSS-Oral-Cancer-Detection-Screening-System
```

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

## Deploy to Hugging Face Spaces

The repository includes a Docker setup for Hugging Face Spaces. It uses Python 3.11 and CPU-only PyTorch wheels, serves the app on port `7860`, and excludes datasets and local reports from the upload. Hugging Face currently requires a PRO subscription or paid compute for Docker Spaces running on basic CPU hardware; check your account's plan before deployment.

1. Install the Hugging Face Hub tools with `python -m pip install huggingface_hub`, then authenticate locally with a **write** token using `hf auth login`. Keep the token private; do not add it to this repository.
2. Create a new Space ID under your Hugging Face account, such as `your-account/oocss-screening-demo`.
3. From the project directory, run `python -m pip install huggingface_hub`, then `python deploy_to_hf.py your-account/oocss-screening-demo`.
4. Wait for the Docker build to finish on the Space page. The health endpoint is `/health`.

The Space is public so anyone can open the demo. Uploaded images and optional risk factors are sent to the hosted server for inference; this app does not retain them after each request. Do not submit identifiable or private patient data. This project is an academic prototype and is not a diagnostic service.

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
