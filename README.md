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

## 📊 Methodology & Mathematical Formulas (Paper 2 Alignment)

### 1. Segmentation & Damage Area (Section 5.1 & Page 20)
- **Contour Area:** Pothole contour area calculated via mask polygons:
  $$\text{Area}_{\text{pothole}} = \oint \text{contour}(x, y) \, dx\,dy \quad (\text{pixels})$$
- **Total Damage Percentage:**
  $$\text{Percentage of Damage (\%)} = \left(\frac{\sum \text{Area}_{\text{pothole}}}{\text{Total Image Area}}\right) \times 100$$

### 2. Relative Pothole Depth (RPD) (Section 5.1 & Page 21)
- **Pothole Depth ($PD$):** Mean normalized depth within the segmented pothole mask ($0.0 \le PD \le 1.0$).
- **Surrounding Depth ($SD$):** Mean depth restricted strictly to immediate surrounding road pixels (adjacent ring boundary).
- **Cavity Validation:** Verifies that $PD > SD$ (confirms recessed cavity).
- **Relative Pothole Depth Formula:**
  $$\text{RPD (\%)} = \frac{PD - SD}{SD} \times 100$$
- **Absolute Depth Difference:**
  $$\Delta D = (PD - SD) \times 100$$

### 3. Hazard Severity Classification
Based on RPD and Area:
- **Low:** $\text{RPD} < 5\%$
- **Moderate:** $5\% \le \text{RPD} < 12\%$
- **High:** $12\% \le \text{RPD} < 18\%$ or $\text{Area} > 4000\text{ px}$ with $\text{RPD} \ge 6\%$
- **Severe:** $\text{RPD} \ge 18\%$ or $\text{Area} > 8000\text{ px}$ with $\text{RPD} \ge 10\%$

