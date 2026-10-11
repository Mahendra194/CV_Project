# YOLOv8-PD-Seg: Changelog & Documentation
## Kya Kiya, Kaise Kiya, Kyun Kiya

---

## 📌 Project Overview

**Original Paper (Paper 1):** YOLOv8-PD — Road damage **detection** (bounding box only)  
**Reference Paper (Paper 2):** Pothole Characterization — **Segmentation + depth estimation**  
**Our Model:** YOLOv8-PD-Seg — Paper 1 ke improvements + Paper 2 ki segmentation = **best of both**

---

## 🔄 Changes Made

### 1. Project Structure Created

```
CV_Project/
├── models/
│   ├── __init__.py              # Package init
│   ├── modules.py               # Custom modules (BOT, LSKA, C2fGhost, Proto, LSCD)
│   └── yolov8_pd_seg.py         # Full model definition
├── configs/
│   └── yolov8-pd-seg.yaml       # Model architecture config
├── datasets/                    # Dataset storage (auto-created)
├── results/
│   └── predictions/             # Inference results
├── docs/
│   └── CHANGES.md               # This file
├── prepare_dataset.py           # Download & prepare RDD2022
├── train.py                     # Training script (baseline + improved)
├── val.py                       # Validation & comparison
├── predict.py                   # Inference with visualization
└── requirements.txt             # Python dependencies
```

### 2. Custom Modules Implemented (`models/modules.py`)

#### From Paper 1 (YOLOv8-PD):

| Module | What it does | Why |
|--------|-------------|-----|
| **GhostModule** | Cheap linear ops se extra features banata hai | Computational cost ~50% kam |
| **C2fGhost** | C2f block with GhostBottleneck | Lightweight neck — params kam |
| **BOTModule** | Bottleneck Transformer — MHSA + CNN | Global context — lambi cracks detect |
| **LSKAModule** | Large Separable Kernel Attention | Large receptive field — road damage ke liye |

#### From Paper 2 (Pothole Characterization) — NEW:

| Module | What it does | Why |
|--------|-------------|-----|
| **ProtoModule** | 32 prototype masks generate karta hai | Instance segmentation ke liye "building blocks" |
| **LSCDSegHead** | LSCD-Head + mask coefficient branch | Har detection ke saath 32 mask coefficients predict |

### 3. Full Model Architecture (`models/yolov8_pd_seg.py`)

```
Input Image (640×640)
        │
        ▼
┌─── BACKBONE ───────────────────┐
│  Stem → Conv layers            │
│  Stage 1-3 → C2fGhost blocks  │  ← Paper 1
│  Stage 4 → BOT Module         │  ← Paper 1 (Transformer)
│  Backbone LSKA                │  ← Paper 1 (Attention)
└────────────────────────────────┘
        │
        ▼
┌─── NECK ───────────────────────┐
│  FPN (top-down) + C2fGhost    │  ← Paper 1
│  PAN (bottom-up) + C2fGhost   │  ← Paper 1
│  LSKA at each scale           │  ← Paper 1
└────────────────────────────────┘
        │
        ▼
┌─── HEAD ───────────────────────┐
│  LSCD-Seg-Head at P3, P4, P5  │  
│  ├── cls_head → class pred    │  ← Paper 1
│  ├── reg_head → bbox pred     │  ← Paper 1
│  └── mask_head → 32 coeffs   │  ← Paper 2 (NEW!)
│                                │
│  Proto Module (on P3)          │  ← Paper 2 (NEW!)
│  └── 32 prototype masks       │
└────────────────────────────────┘
        │
        ▼
Output: Boxes + Classes + Masks
```

### 4. Training Pipeline (`train.py`)

Three training modes:
- **`--model baseline`**: Vanilla YOLOv8n-seg (no improvements)
- **`--model improved`**: YOLOv8n-seg with tuned hyperparameters for road damage
- **`--model custom`**: Full custom architecture from `yolov8_pd_seg.py`

Key training differences from baseline:
- Lower LR (0.0008 vs 0.001) — transformer components need stability
- Longer warmup (5 epochs vs 3) — BOT module needs gradual learning
- Enhanced augmentation: rotation, shear, copy-paste — road damage diversity
- `copy_paste=0.1` — augments small damages by pasting them on other images

### 5. Evaluation Pipeline (`val.py`)

Metrics computed:
- **Box mAP@0.5** — bounding box detection (same as Paper 1)
- **Box mAP@0.5:0.95** — strict bbox metric
- **Mask mAP@0.5** — segmentation quality (NEW — Paper 2 capability)
- **Mask mAP@0.5:0.95** — strict segmentation metric
- Per-class metrics for all 4 damage types

### 6. Inference Pipeline (`predict.py`)

Features:
- Box + mask overlay on images
- Per-detection: class, confidence, area in pixels
- Side-by-side comparison mode (`--compare`)
- Color-coded by damage type

---

## 📊 How to Run (Step by Step)

### Step 1: Install Dependencies
```bash
cd /Users/mahendrajajra/Desktop/CV_Project
pip install -r requirements.txt
```

### Step 2: Prepare Dataset
```bash
python prepare_dataset.py
```

### Step 3: Train Both Models (for comparison)
```bash
# Train baseline (no improvements)
python train.py --model baseline --epochs 100

# Train improved (with all our improvements)
python train.py --model improved --epochs 100

# Or train both at once
python train.py --model both --epochs 100

# Quick test (2 epochs — just to check everything works)
python train.py --model both --epochs 2
```

### Step 4: Compare Results
```bash
python val.py --compare
```

### Step 5: Run Inference
```bash
python predict.py --source test_images/ --compare
```

---

## 🧪 Expected Improvements

| Metric | Baseline (YOLOv8n-seg) | Our Model (YOLOv8-PD-Seg) |
|--------|----------------------|--------------------------|
| Box mAP@0.5 | ~52-54% | ~55-57% (+2-3%) |
| Mask mAP@0.5 | ~48-50% | ~51-54% (+2-3%) |
| Parameters | ~3.4M | ~2.8M (-18%) |
| Speed (FPS) | ~130 | ~110-120 |

---

## 📚 References

1. **Paper 1:** Zeng, J. & Zhong, H. (2024). "YOLOv8-PD: an improved road damage detection algorithm based on YOLOv8n model." Scientific Reports, 14, 12052.
2. **Paper 2:** Baroudi, U., BaHamid, A., Elalfy, Y., & Al Alami, Z. (2025). "Enhancing Pothole Detection and Characterization: Integrated Segmentation and Depth Estimation in Road Anomaly Systems." arXiv:2504.13648.
3. **GhostNet:** Han, K. et al. (2020). "GhostNet: More Features from Cheap Operations." CVPR 2020.
4. **BoTNet:** Srinivas, A. & Lin, T.Y. (2021). "Bottleneck Transformers for Visual Recognition." CVPR 2021.
5. **LSKA:** Li, Y. et al. (2023). "Large Selective Kernel Network for Remote Sensing Object Detection."
