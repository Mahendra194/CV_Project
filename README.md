# Road Damage Detection & Pothole Characterization (YOLOv8-PD-Seg)

An integrated deep learning framework combining **lightweight damage detection/segmentation** with **monocular depth estimation** for road anomaly characterization (bounding boxes, instance segmentation masks, depth extraction, and volume estimation).

---

## 📌 Overview

This project unifies and advances research from two recent papers:
1. **Paper 1 (YOLOv8-PD):** *"YOLOv8-PD: an improved road damage detection algorithm based on YOLOv8n model"* (Scientific Reports, 2024)
   - Incorporates **BOT Module** (Bottleneck Transformer) for global crack context.
   - Utilizes **LSKA** (Large Separable Kernel Attention) for wide receptive field with minimal parameters.
   - Integrates **C2fGhost** to cut computational redundancy in the neck.
   - Employs **LSCD** (Lightweight Shared Convolution) head design.
2. **Paper 2 (Pothole Characterization):** *"Enhancing Pothole Detection and Characterization: Integrated Segmentation and Depth Estimation in Road Anomaly Systems"* (2025)
   - Extends detection to pixel-level **Instance Segmentation** (Proto Module & Mask branch).
   - Integrates **Monocular Depth Estimation** to extract relative pothole depth and estimate damage volume.

---

## 📁 Repository Structure

```
├── configs/
│   └── yolov8-pd-seg.yaml        # YOLOv8-PD-Seg architecture specification
├── models/
│   ├── modules.py                # BOT, LSKA, C2fGhost, Proto, and LSCDSegHead modules
│   ├── depth_estimator.py        # Monocular depth estimation & pothole depth extraction
│   └── yolov8_pd_seg.py          # PyTorch model definitions
├── prepare_dataset.py            # Converts RDD2022 bounding boxes to segmentation polygons
├── register_modules.py           # Hooks custom modules into Ultralytics YOLOv8 engine
├── train.py                      # Multi-model training pipeline (baseline / improved / custom)
├── val.py                        # Validation and mAP evaluation script
├── predict.py                    # Inference on images and videos
├── characterize_potholes.py      # End-to-end pipeline: Segmentation + Depth Map + Volume estimation
├── compare_models.py             # Benchmark & comparison utility
└── requirements.txt              # Python dependencies
```

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/Mahendra194/CV_Project.git
cd CV_Project
pip install -r requirements.txt
```

### 2. Dataset Preparation (RDD2022)

```bash
python prepare_dataset.py
```

### 3. Training

```bash
# Train the improved custom model (YOLOv8-PD-Seg)
python train.py --model custom --epochs 50 --batch 16
```

### 4. Pothole Characterization (Segmentation + Depth)

Run the full integrated pipeline on any test image to generate instance masks, depth maps, and volume metrics:

```bash
python characterize_potholes.py \
  --image test_images/sample.jpg \
  --model runs/segment/custom_yolov8_pd_seg/weights/best.pt \
  --output result.jpg
```

---

## 📊 Methodology Summary

```
Input Image (640x640)
       │
       ▼
[Improved Backbone (Paper 1)]
  Conv → C2fGhost → SPPF → BOTModule (Global Self-Attention)
       │
       ▼
[Improved Neck (Paper 1)]
  C2fGhost + LSKA Attention (Large Receptive Field)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[LSCD-Seg Head]                 [Depth Estimator (Paper 2)]
  ├── Classification              ├── Monocular Depth Map
  ├── Bounding Boxes              └── Surface Reference Baseline
  └── Prototype Masks (Proto)
       │                                 │
       └────────────────┬────────────────┘
                        ▼
       [Pothole Characterization]
         • Area (Pixel Count)
         • Depth (Relative Elevation Difference)
         • Volume Estimation (Area × Relative Depth)
```
