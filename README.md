# OOCSS — Oral Cancer Detection Screening System

OOCSS is an academic research prototype for oral-cavity image classification. It combines a ResNet-50 image classifier with a Random Forest model that can use optional clinical factors to produce a risk-stratification result.

> **Research use only.** This project is not a medical device and does not diagnose cancer. Its predictions and recommendations must not guide care or replace assessment by a qualified clinician.

## Availability

The source code and trained model files are published in this GitHub repository. The web app runs locally or in a Docker container; **there is no public hosted demo at this time**.

## Features

- Classifies an image into **Normal**, **Benign**, **OPMD**, or **OSCC**.
- Shows the model's predicted class and class probabilities.
- Optionally combines the image probabilities with clinical factors for a research risk estimate.
- Includes a browser interface, a FastAPI endpoint, and a command-line prediction script.

## Requirements

- Python **3.10 or 3.11** for local installation. The pinned PyTorch packages in `requirements.txt` are intended for these versions.
- Git LFS to download the trained ResNet-50 weights from this repository.
- Internet access during setup to install packages. Training also downloads the standard ImageNet ResNet-50 checkpoint if it is not cached; running inference with the included trained weights does not need that checkpoint.

## Download the project and model

Install [Git LFS](https://git-lfs.com/) before cloning. Then run:

```bash
git lfs install
git clone https://github.com/ijayslay/OOCSS-Oral-Cancer-Detection-Screening-System.git
cd OOCSS-Oral-Cancer-Detection-Screening-System
```

The trained files are `models/resnet50_oocss.pth` and `models/random_forest_oocss.joblib`.

## Run the web app locally

From the project directory, create a virtual environment and install the dependencies.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser. The health endpoint is `/health`; the interactive API documentation is at `/docs`.

## Run with Docker

If Docker is installed, build and start the local container:

```bash
docker build -t oocss .
docker run --rm -p 7860:7860 oocss
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860). The container uses CPU-only PyTorch and does not include the training dataset or local output files.

## Command-line prediction

Run a prediction on one image:

```bash
python predict.py --image path/to/oral_photo.jpg
```

Optional clinical factors can be supplied to the risk model:

```bash
python predict.py --image path/to/oral_photo.jpg --age 52 --sex 1 --tobacco 2 --alcohol 1 --betel 2 --duration 8 --size 12.5
```

Use `python predict.py --help` to see the accepted inputs. Do not provide identifiable patient information.

## Model training

Pre-trained project weights are included, so training is not required to run the web app. For training or fine-tuning, arrange the dataset as follows:

```text
data/
├── train/{Normal,Benign,OPMD,OSCC}/
├── val/{Normal,Benign,OPMD,OSCC}/
└── test/{Normal,Benign,OPMD,OSCC}/
```

Then run:

```bash
python test_setup.py
python train.py
```

You can also configure Kaggle credentials and use `python download_dataset.py`. Dataset files are excluded from Git because of their size and licensing requirements. Check the dataset license and terms before downloading, sharing, or training on it. Training writes model files to `models/` and plots and history files to `outputs/`; training may take several hours on a CPU.

## Project layout

```text
app.py                 FastAPI web application
predict.py             Single-image command-line prediction
train.py               Training entry point
download_dataset.py    Optional Kaggle dataset downloader
config.py              Model, data, and training configuration
models/                Trained model artifacts
static/                Browser interface
utils/                 Dataset, model, training, evaluation, and risk code
```

## Privacy and responsible use

The app accepts oral-cavity images and optional clinical factors. When you run it locally, requests are handled by the app on your computer. The app does not retain submitted images or factors after processing. Do not upload identifiable or private patient data.

Model results can vary with image quality, population, equipment, and dataset composition. A predicted probability or risk category is not a diagnosis and must never be used to delay professional assessment. Any concerning oral lesion should be evaluated by a qualified healthcare professional. No clinical or regulatory validation is implied.

Before redistributing this project, confirm that you have permission to share the model weights, dataset-derived outputs, and any included materials.
