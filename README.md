# Road Damage Detection & Pothole Characterization (YOLOv8-PD-Seg)

An integrated deep learning framework combining **lightweight damage detection and instance segmentation** with **monocular depth estimation** for comprehensive road anomaly characterization (bounding boxes, polygon instance masks, 3D relative depth $RPD\%$, and automated hazard severity assessment).

Authors: Divyam and Mahendra

---

## Overview

This project advances road surface inspection by combining:
1. **Base Paper Architecture (YOLOv8-PD):** *"YOLOv8-PD: an improved road damage detection algorithm based on YOLOv8n model"* (Scientific Reports, 2024)
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

## Project Architecture

YOLOv8-PD-Seg is a unified multi-task vision framework that integrates lightweight instance segmentation with monocular depth estimation to detect, segment, and quantitatively characterize road surface anomalies.

```
                              Input Road Image (640 x 640 x 3)
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               BACKBONE NETWORK (Lightweight)                           │
│  Standard Convolutions + C2f Feature Extraction                                       │
│  ├── BOTModule (BoTNet MHSA): Captures long-span global crack continuity               │
│  └── LSKAModule: Large Separable Kernel Attention (19x19 Receptive Field)              │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Multi-scale Features (P3, P4, P5)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              NECK FEATURE FUSION (PANet / FPN)                         │
│  ├── Multi-scale Feature Pyramid Aggregation                                           │
│  └── C2fGhost Blocks: Ghost convolutions cut parameter count by 12.5% and ~50% FLOPs   │
└───────────────────────┬───────────────────────────────────────┬────────────────────────┘
                        │ P3, P4, P5 Features                   │ Raw Image / P3 Feature
                        ▼                                       ▼
┌───────────────────────────────────────────────┐   ┌────────────────────────────────────┐
│      INSTANCE SEGMENTATION BRANCH (LSCD-Seg)   │   │    MONOCULAR DEPTH BRANCH (3D)     │
│                                               │   │                                    │
│  ├── ProtoModule (P3 Scale)                   │   │  DenseDepth / MiDaS Depth Engine   │
│  │   └── 32 Prototype Masks (160 x 160)       │   │  ├── Dense Feature Encoder         │
│  │                                            │   │  └── Bilinear Upsampling Decoder   │
│  └── LSCD-Seg Shared Convolution Head         │   │                                    │
│      ├── Class Branch: D00, D10, D20, D40     │   │  Output: Normalized Depth Map      │
│      ├── Bounding Box Branch: (x, y, w, h)    │   │  D(x, y) in [0, 1] (Continuous)    │
│      └── Mask Branch: 32 Mask Coefficients    │   │                                    │
│                                               │   └─────────────────┬──────────────────┘
│  Linear Combination: Sigmoid(Coeffs x Protos) │                     │
│  Output: Pixel-Level Segmentation Masks       │                     │
└───────────────────────┬───────────────────────┘                     │
                        │                                             │
                        │   Polygon Mask M(x, y)                      │ Depth Map D(x, y)
                        └───────────────────────┬─────────────────────┘
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      3D POTHOLE CHARACTERIZATION & HAZARD REASONING                    │
│                                                                                        │
│  1. Damage Geometry:        Contour Area A = ∮ contour ds, Road Damage % = (A / WxH)x100│
│  2. Pothole Depth (PD):     Mean depth inside damage mask: PD = (1/N) ∑ D(x, y)        │
│  3. Surrounding Depth (SD): Mean depth of adjacent asphalt ring: SD = (1/M) ∑ D(x, y) │
│  4. Cavity Verification:    Confirms physical cavity depression if PD > SD (Stain Filter)│
│  5. Relative Depth (RPD):   RPD = ((PD - SD) / SD) x 100%                              │
│  6. Hazard Rating:          Automated classification: SEVERE, HIGH, MODERATE, LOW      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
                       4-Panel Integrated Diagnostic HUD Display
            [Panel A: Image | Panel B: Mask | Panel C: Depth | Panel D: HUD]
```

### Architectural Modules Overview

| Component | Architecture | Role & Technical Impact |
| :--- | :--- | :--- |
| **BOTModule** | Bottleneck Transformer (Multi-Head Self-Attention) | Replaces standard convolutions in deep backbone layers. Computes non-local interactions across the full image to capture continuous crack topologies regardless of shadow or road markings. |
| **LSKAModule** | Large Separable Kernel Attention ($19\times19$ kernel) | Decomposes 2D large convolution into cascaded 1D horizontal and vertical depth-wise operations, providing a massive effective receptive field without parameter explosion. |
| **C2fGhost** | GhostBottleneck Feature Extraction | Replaces heavy C2f blocks in the feature fusion neck. Generates intrinsic feature maps with cheap linear transforms, cutting parameters by 12.5% and neck FLOPs by ~50%. |
| **ProtoModule** | Mask Prototype Generator (P3 feature scale) | Generates 32 high-resolution ($160 \times 160$) base prototype masks using separable convolutions, providing high-fidelity spatial details. |
| **LSCD-Seg Head** | Lightweight Shared Convolution Detection & Segmentation | Extends the base paper's detection head with an efficient third branch producing 32 mask coefficients per instance, multiplied with prototype masks to yield exact damage boundaries. |
| **Depth Engine** | Monocular Depth Estimation (DenseDepth / MiDaS) | Predicts per-pixel dense relative elevation maps $D(x, y) \in [0, 1]$ directly from single-lens monocular images without requiring expensive stereo cameras or LiDAR. |
| **Characterization** | Relative Pothole Depth ($RPD$) & Hazard Logic | Computes physical depression metrics ($PD$, $SD$), validates genuine cavities ($PD > SD$), and grades road hazards into 4 actionable maintenance tiers. |

---

## Model Performance Benchmark

### Overall Model Performance on RDD2022

| Model Architecture | Task / Modality | mAP@50 (%) | mAP@50-95 (%) | Parameters (M) | GFLOPs | Inference Speed (FPS) | Output Capabilities |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8n (Standard Baseline)** | Object Detection | 74.5% | 51.2% | 3.2 M | 8.7 | 142 FPS | 2D Bounding Boxes only |
| **YOLOv8n-seg (Standard Baseline)** | Instance Segmentation | 77.2% | 53.6% | 3.4 M | 8.9 | 128 FPS | 2D Bounding Boxes + Segmentation Masks |
| **Base Paper (YOLOv8-PD)** | Object Detection + Attention | 82.0% | 58.4% | 3.1 M | 7.6 | 135 FPS | 2D Bounding Boxes with attention |
| **Our Model (PD-Seg + Depth + HUD)** | **Unified Segmentation + 3D Depth** | **87.0%** | **63.1%** | **3.9 M** | **9.2** | **105 FPS** | **Polygonal Masks + 3D Depth + RPD% + Severity HUD** |
| **Net Gain vs. Baseline (YOLOv8n)** | *Overall Improvement* | **+12.5%** | **+11.9%** | *Lightweight* | *Real-time* | *Dashcam Ready* | *Multi-Modal Road Hazard Suite* |
| **Net Gain vs. Base Paper (YOLOv8-PD)** | *Architectural Upgrade* | **+5.0%** | **+4.7%** | *+0.8 M params* | *+1.6 GFLOPs* | *>100 FPS* | *Adds Instance Seg + 3D Depth + RPD Verification* |

### Class-Wise Detection & Segmentation Performance (mAP@50)

| Damage Class Code | Damage Description | Baseline YOLOv8n-seg | Base Paper (YOLOv8-PD) | Our Model (YOLOv8-PD-Seg) | Relative Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **D00** | Longitudinal Crack | 78.2% | 82.8% | **88.1%** | +5.3% |
| **D10** | Transverse Crack | 75.8% | 81.2% | **85.9%** | +4.7% |
| **D20** | Alligator (Mesh) Crack | 78.9% | 83.6% | **89.2%** | +5.6% |
| **D40** | Pothole (Cavity) | 75.9% | 80.4% | **84.8%** | +4.4% |
| **Mean (mAP@50)** | **All Classes** | **77.2%** | **82.0%** | **87.0%** | **+5.0%** |

---

## Getting Started

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
python characterize_potholes.py --image assets/sample_images/1_Dual_Pothole_Input.png
```

This generates a 4-panel diagnostic visualization:
1. **Panel A:** Input Road Image
2. **Panel B:** Instance Segmentation Masks
3. **Panel C:** Monocular Depth Map
4. **Panel D:** Characterization HUD (Area, $PD, SD, RPD\%$, and Hazard Rating)

---

## Research Paper & Detailed Reports

For complete presentation-ready visual reports and detailed mathematical derivations, see:
* **[latex_paper/](latex_paper/)**: Overleaf-ready IEEE conference research paper source (`main.tex`) with figures and compilation guide.
* **[showcase.md](showcase.md)**: Full visual report covering all tested road scenes and metrics.
* **[docs/base_architecture_and_changes.md](docs/base_architecture_and_changes.md)**: Exhaustive architectural documentation of the Base Paper and all enhancements.
* **[docs/project_improvements.md](docs/project_improvements.md)**: Research foundation and equations.
