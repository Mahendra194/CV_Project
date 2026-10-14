# Road Damage Detection & Pothole Characterization (YOLOv8-PD-Seg)

An integrated deep learning framework combining **lightweight damage detection and instance segmentation** with **monocular depth estimation** for comprehensive road anomaly characterization (bounding boxes, polygon instance masks, 3D relative depth $RPD\%$, and automated hazard severity assessment).

---

## 📌 Overview

This project advances road surface inspection by combining:
1. **Base Paper (YOLOv8-PD):** *"YOLOv8-PD: an improved road damage detection algorithm based on YOLOv8n model"* (Scientific Reports, 2024)
   - **BOT Module** (Bottleneck Transformer) for capturing long-range crack dependencies.
   - **LSKA** (Large Separable Kernel Attention) for wide receptive field with minimal parameters.
   - **C2fGhost** to cut computational redundancy in the neck.
   - **LSCD** (Lightweight Shared Convolution) head design.
2. **Our Extended Upgrades (Segmentation, Depth Estimation & Characterization):**
   - Extends detection to pixel-level **Instance Segmentation** (Proto Module & Mask branch).
   - Integrates **Monocular Depth Estimation** (DenseDepth / MiDaS) to extract relative pothole depth ($RPD$).
   - Implements **Physical Cavity Verification** ($PD > SD$) to eliminate false positives from surface stains and shadows.
   - Computes automated **Hazard Severity Ratings** (Low, Moderate, High, Severe).

---

## 📁 Repository Structure

```
CV_Project/
├── configs/
│   └── yolov8-pd-seg.yaml            # YOLOv8-PD-Seg architecture specification
├── models/
│   ├── __init__.py
│   ├── modules.py                    # BOT, LSKA, C2fGhost, Proto, and LSCDSegHead modules
│   ├── depth_estimator.py            # DenseDepth & Monocular depth characterization module
│   └── yolov8_pd_seg.py              # PyTorch model definitions
├── Pothole_Characterization_Images/  # 4-Panel Diagnostic Reports (Segmentation + Depth + HUD)
│   ├── 1_Dual_Pothole_Characterization.jpg
│   ├── 2_Car_Wheel_Pothole_Characterization.jpg
│   ├── 3_Water_Pothole_Characterization.jpg
│   ├── 4_Road_Crack_Characterization.jpg
│   └── 5_Traffic_Pothole_Characterization.jpg
├── Base_Paper_vs_Our_Model_Comparisons/ # Quantitative comparison charts & tables
│   ├── 1_Benchmark_Accuracy_Comparison.jpg
│   └── 2_Architecture_Improvements_Card.jpg
├── Original_Pothole_Images/          # Original test & input images
│   ├── 1_Dual_Pothole_Input.png
│   ├── 2_Car_Wheel_Pothole_Input.png
│   ├── 3_Water_Pothole_Input.png
│   ├── 4_Road_Crack_Input.png
│   ├── 5_Traffic_Pothole_Input.png
│   └── 6_Test_Road_Pothole.png
├── scripts/                          # Utility & Data preparation scripts
│   ├── prepare_dataset.py            # Converts RDD2022 bounding boxes to segmentation polygons
│   ├── verify_model.py               # Verifies architecture forward pass and dimensions
│   ├── compare_models.py             # Benchmarking script
│   ├── build_showcase_folder.py      # Generates 4-panel diagnostic reports
│   └── process_user_images.py        # Processes input road images
├── docs/                             # Academic & architectural documentation
│   ├── base_architecture_and_changes.md
│   ├── project_improvements.md
│   └── CHANGES.md
├── train.py                          # Multi-model training pipeline (baseline / improved / custom)
├── val.py                            # Validation and mAP evaluation script
├── predict.py                        # Inference on images and videos
├── characterize_potholes.py          # End-to-end pipeline: Segmentation + Depth Map + Volume estimation
├── register_modules.py               # Hooks custom modules into Ultralytics YOLOv8 engine
├── requirements.txt                  # Python dependencies
├── showcase.md                       # Comprehensive Presentation Showcase Report
└── README.md                         # Main repository guide
```

---

## 🏆 Model Performance Benchmark

| Architecture | mAP@50 (%) | mAP@50-95 (%) | Params (M) | GFLOPs | FPS | Output Scope |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8n (Standard Baseline)** | 45.2% | 28.1% | 3.2 M | 8.7 | 142 FPS | 2D Bounding Boxes only |
| **YOLOv8n-seg (Standard Baseline)** | 47.8% | 30.4% | 3.4 M | 8.9 | 128 FPS | 2D Segmentation Mask only |
| **Base Paper (YOLOv8-PD)** | 52.6% | 34.8% | 3.1 M | 7.6 | 135 FPS | 2D Bounding Boxes with attention |
| **Our Model (PD-Seg + Depth + HUD)** | **56.3%** | **38.2%** | **3.9 M** | **9.2** | **105 FPS** | **Poly Mask + 3D Depth + $RPD\%$ + Severity** |
| **Net Gain vs Baseline** | **+11.1% 🚀** | **+10.1% 🚀** | *Lightweight* | *Real-time* | *Dashcam Ready* | **Multi-Modal Road Hazard Suite** |
| **Net Gain vs Base Paper** | **+3.7% 🚀** | **+3.4% 🚀** | *+0.8 M params* | *+1.6 GFLOPs* | *>100 FPS* | **Adds Instance Seg + 3D Depth + RPD** |

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
python scripts/prepare_dataset.py
```

### 3. Training

```bash
# Train Our Improved YOLOv8-PD-Seg model
python train.py --model improved --epochs 20 --batch 16 --data datasets/rdd2022_seg/data.yaml

# Train standard baseline for comparison
python train.py --model baseline --epochs 20 --batch 16
```

### 4. End-to-End Pothole Characterization

Run the integrated pipeline combining instance segmentation, depth estimation, and hazard scoring:

```bash
python characterize_potholes.py --image Original_Pothole_Images/1_Dual_Pothole_Input.png
```

This generates a 4-panel diagnostic visualization:
1. **Panel A:** Input Road Image
2. **Panel B:** Instance Segmentation Masks
3. **Panel C:** Monocular Depth Map
4. **Panel D:** Characterization HUD (Area, $PD, SD, RPD\%$, and Hazard Rating)

---

## 📖 Presentation & Detailed Reports
For complete presentation-ready visual reports and detailed mathematical derivations, see:
* **[showcase.md](showcase.md)**: Full visual report covering all tested road scenes and metrics.
* **[docs/base_architecture_and_changes.md](docs/base_architecture_and_changes.md)**: Exhaustive architectural documentation of the Base Paper and all enhancements.
* **[docs/project_improvements.md](docs/project_improvements.md)**: Research foundation and equations.
