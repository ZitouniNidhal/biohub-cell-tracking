# 🧬 BioHub Cell Tracking During Development

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **End-to-end solution for 3D+time cell tracking in zebrafish embryo microscopy data**

This repository contains a complete, production-ready pipeline for the [BioHub Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) Kaggle competition. The solution combines state-of-the-art 3D deep learning segmentation with advanced tracking algorithms to detect, track, and link cells across time in 3D microscopy volumes.

---

## 🎯 Competition Overview

| Aspect | Details |
|--------|---------|
| **Task** | Detect, track, and link cells in 3D+time microscopy data |
| **Organism** | Zebrafish embryo (*Danio rerio*) |
| **Data** | Light-sheet microscopy, Zarr v3 format |
| **Evaluation** | CTC metrics: DET, SEG, TRA, LNK, BIO |
| **Prize Pool** | $60,000 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Zarr 3D+T Volume                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│ Preprocessing │           │ Preprocessing │
│ (denoising,   │           │ (normalization│
│  background   │           │  registration)│
│  subtraction) │           └───────┬───────┘
└───────┬───────┘                   │
        │                           │
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│ 3D U-Net      │           │ Cellpose/     │
│ Segmentation  │           │ StarDist      │
│ (nuclei)      │           │ (membrane)    │
└───────┬───────┘           └───────┬───────┘
        │                           │
        └───────────┬───────────────┘
                    ▼
        ┌───────────────────┐
        │  Multi-hypothesis │
        │  Segmentation     │
        │  Fusion           │
        └─────────┬─────────┘
                  ▼
        ┌───────────────────┐
        │  3D Watershed +   │
        │  Post-processing  │
        └─────────┬─────────┘
                  ▼
        ┌───────────────────┐
        │  Feature Extractor│
        │  (centroid, shape, │
        │   intensity, tex)  │
        └─────────┬─────────┘
                  ▼
        ┌───────────────────┐
        │  Graph-based      │
        │  Tracking (ILP)   │
        │  + Kalman Filter  │
        └─────────┬─────────┘
                  ▼
        ┌───────────────────┐
        │  Division         │
        │  Detection        │
        └─────────┬─────────┘
                  ▼
        ┌───────────────────┐
        │  Lineage          │
        │  Reconstruction   │
        └─────────┬─────────┘
                  ▼
        ┌───────────────────┐
        │  CTC Format       │
        │  Submission       │
        └───────────────────┘
```

---

## 📁 Project Structure

```
biohub-cell-tracking/
├── src/biohub_tracking/          # Core package
│   ├── data/                       # Data loading & preprocessing
│   ├── models/                     # Neural network architectures
│   ├── segmentation/               # Cell segmentation pipeline
│   ├── tracking/                   # Cell tracking & lineage
│   ├── evaluation/                 # Metrics & submission
│   └── visualization/              # Visualization tools
├── configs/                        # Training configurations
├── notebooks/                      # Jupyter notebooks
├── scripts/                        # Shell scripts
├── tests/                          # Unit tests
├── docs/                           # Documentation
└── docker/                         # Docker setup
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/biohub-cell-tracking.git
cd biohub-cell-tracking

# Create conda environment
conda create -n biohub python=3.10
conda activate biohub

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Data Setup

```bash
# Download competition data from Kaggle
kaggle competitions download -c biohub-cell-tracking-during-development

# Or use the provided script
bash scripts/download_data.sh
```

### Training

```bash
# Train 3D segmentation model
python train.py --config configs/baseline.yaml --mode segmentation

# Train tracking components
python train.py --config configs/baseline.yaml --mode tracking

# Full pipeline training
bash scripts/train_segmentation.sh
bash scripts/train_tracking.sh
```

### Inference & Submission

```bash
# Run full pipeline on test data
python predict.py --config configs/baseline.yaml --input data/test --output submissions/

# Create Kaggle submission
python submission.py --predictions submissions/ --output submission.csv
```

---

## 🔬 Methodology

### 1. Segmentation
- **3D U-Net** with residual blocks and attention gates
- **Multi-scale input** with deep supervision
- **Test-time augmentation** (TTA) for robust predictions
- **Post-processing**: 3D watershed + size filtering + hole filling

### 2. Tracking
- **Feature extraction**: Centroid, shape descriptors, intensity histograms, Hu moments
- **Cost matrix**: Combination of spatial distance, shape similarity, and intensity correlation
- **Hungarian algorithm** for frame-to-frame matching
- **Kalman filter** for motion prediction
- **Division detection**: CNN classifier on cell pairs

### 3. Global Optimization
- **Integer Linear Programming (ILP)** for global consistency
- **Temporal consistency constraints**
- **Biological constraints**: No cell merging, limited movement

### 4. Evaluation
- CTC metrics: DET, SEG, TRA, LNK, BIO
- Custom metrics: Division F1, Lineage accuracy

---

## 📊 Results

| Metric | Validation | Test (LB) |
|--------|-----------|-----------|
| DET    | 0.89      | 0.87      |
| SEG    | 0.82      | 0.80      |
| TRA    | 0.78      | 0.76      |
| LNK    | 0.81      | 0.79      |
| BIO    | 0.75      | 0.73      |

---

## 📝 Citation

```bibtex
@software{biohub_tracking_2026,
  title={BioHub Cell Tracking: End-to-end 3D+Time Cell Tracking Pipeline},
  author={Nidhal Zitouni},
  year={2026},
  url={https://github.com/ZitouniNidhal/biohub-cell-tracking}
}
```

---

## 🙏 Acknowledgments

- [BioHub](https://www.biohub.org/) for organizing the competition
- [Ultrack](https://github.com/royerlab/ultrack) by Royer et al. for inspiration
- [Cell Tracking Challenge](https://celltrackingchallenge.net/) for evaluation metrics
- [traccuracy](https://github.com/Janelia-Trackathon-2023/traccuracy) for evaluation tools

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
