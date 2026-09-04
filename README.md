# ☀️ SAPTASHVA
**Solar Flare Precursor Detection & Early-Warning AI System**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.14-red?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Accuracy: 95.35%](https://img.shields.io/badge/Validation_Accuracy-95.35%25-brightgreen?logo=googleanalytics&logoColor=white)](#performance-milestones)

**SAPTASHVA** is a high-performance, deep-learning pipeline designed to ingest raw space weather telemetry and predict chaotic solar flare eruptions with extreme precision. Built to interface with data from **ISRO's Aditya-L1** (SoLEXS & HEL1OS) and NASA's GOES satellites, SAPTASHVA acts as a critical early-warning AI for space weather anomalies.

---

## 🧠 Core Architecture

SAPTASHVA operates on a two-tier mathematical framework:

### 1. Stage 0: The State Estimator (Deep LSTM)
At its core is a heavily optimized **128x64 Stacked LSTM** neural network. It processes soft and hard X-ray flux derivatives over a 60-minute rolling lookback window to classify the sun's state into four distinct phases:
- 🟢 **Quiet**
- 🟡 **Active**
- 🟠 **Eruptive** (Solar Flare Event)
- 🔵 **Recovery**

### 2. Stage 1: The Escalation Layer (XGBoost)
A tabular XGBoost model sits on top of the LSTM, acting as a secondary logic gate. It implements a strict 3-tier, one-way escalating alert system designed for space mission control rooms:
- 🔭 **Watch** → ⚠️ **Warning** → 🚨 **Alert**

---

## 🚀 Performance Milestones

SAPTASHVA was trained on an enormous dataset consisting of **11 years of continuous Solar Cycle 24 data (2010–2021)**, encompassing over **5 million discrete data points**. 

🏆 **THE 95% BARRIER HAS BEEN OFFICIALLY BROKEN**
Through architectural scaling, physics-based dynamic peak detection labeling, and plateau learning rate schedulers, the model achieved a staggering **95.35% Validation Accuracy** in predicting solar flare states.

| Phase | Architecture | Dataset | Validation Accuracy | Status |
|---|---|---|---|---|
| **Phase 4** | 64x32 LSTM | 260K Samples | 88.00% | Archival |
| **Phase 5** | 64x32 LSTM | 5 Million Samples (11-Year) | 93.67% | Archival |
| **Phase 7** | **128x64 LSTM** | **5 Million Samples (11-Year)** | **95.35%** | **Current Champion** |

---

## ⚙️ Engineering Highlights

- **Spectral Diagnostics**: Includes a 2D Spectrogram deployment module using `astropy` for in-depth Aditya-L1 data visualization.
- **Resilient Compute**: Features dynamic PyTorch disk-checkpointing and graceful degradation to salvage mathematical weights if compute clusters are interrupted mid-epoch.
- **Precision Data Ingestion**: Parses raw `FITS` and `.lc.gz` files, executing perfect microsecond synchronization between Julian Dates (MJD) and Unix timestamps.
- **Cloud Native**: Pre-configured and optimized to leverage A100/T4 GPUs on Google Colab Pro and Vertex AI.

---

## ⚡ Quickstart

We have migrated to **Astral's `uv`** for lightning-fast, highly reliable dependency management.

### 1. Setup the Environment
```bash
# Install dependencies into a new virtual environment instantly
uv venv
uv pip install -r requirements.txt
```

### 2. Run the Neural Engine
```bash
# Train the model on the full cached dataset
uv run python src/stage0/train_goes.py
```

### 3. Run Pipeline Diagnostics
```bash
# End-to-end synthetic pipeline test
uv run python -m tests.test_pipeline

# Real ISRO PRADAN ingestion test
uv run python -m tests.test_ingestion ./data
```

---
*Built as a solo exploration at the bleeding edge of Astrophysics and Deep Learning.*
