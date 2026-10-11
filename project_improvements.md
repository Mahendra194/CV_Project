# 📚 Comprehensive Research, Architecture & Mathematical Foundations
## YOLOv8-PD-Seg: Unified Road Damage Segmentation and Pothole Characterization

This document provides an exhaustive, mathematically rigorous, and architectural breakdown of:
1. **The Base Paper (Paper 1):** *YOLOv8-PD: An improved road damage detection algorithm based on YOLOv8n model* (Scientific Reports, Nature, 2024).
2. **The Secondary Research (Paper 2):** *Enhancing Pothole Detection and Characterization: Integrated Segmentation and Depth Estimation in Road Anomaly Systems* (2025).
3. **Our Unified Novel Architecture (YOLOv8-PD-Seg + Depth Characterization):** The hybrid system synthesizing Paper 1's lightweight attention architecture with Paper 2's instance segmentation and monocular depth estimation pipeline.

---

## 1. Executive Summary & Problem Formulation

### 1.1 Limitations of Traditional Road Distress Detection
Conventional pavement inspection systems rely on standard convolutional object detectors (such as standard YOLOv8n) that output axis-aligned **Bounding Boxes**. In the context of civil engineering and autonomous road survey, bounding boxes present fundamental drawbacks:
* **Severe Spatial Inaccuracy:** Cracks (longitudinal, transverse, alligator) are thin, elongated, and highly irregular. A rectangular bounding box contains up to $80\%\text{--}90\%$ undamaged road asphalt, causing high false-positive background noise.
* **Absence of 3D Cavity Geometry:** Bounding boxes provide zero information regarding the **depth ($PD$)**, **relative elevation ($RPD$)**, or **volumetric displacement** of road depressions. A shallow optical dark spot on asphalt cannot be differentiated from a deep, suspension-damaging pothole.

### 1.2 The Research Objective
To overcome these limitations, we engineered a unified pipeline:
1. **Paper 1 Integration:** Upgrading the feature extraction backbone and neck using **Bottleneck Transformers (BOT)**, **Large Separable Kernel Attention (LSKA)**, and **C2fGhost** modules to capture global crack continuity with minimal computational parameters.
2. **Paper 2 Integration:** Upgrading the detection head to an **Instance Segmentation Head (Proto Module + Mask Coefficients)** and coupling it with a **Monocular Depth Estimation Engine (DenseDepth / Encoder-Decoder)** to extract true pixel area, cavity depression validation ($PD > SD$), relative depth ($RPD$), and damage hazard grading.

---

## 2. IMPROVEMENT 1: Paper 1 Architecture (YOLOv8-PD)

> **Paper 1 Reference:** *Scientific Reports (2024)* — Nature Portfolio  
> **Target:** Lightweight, high-accuracy detection for mobile and edge deployment.

### 2.1 Limitations of Baseline YOLOv8n on Pavement Distress
Standard YOLOv8n utilizes localized $3 \times 3$ convolutional filters throughout its backbone (DarkNet-53 derived) and neck (PANet). Standard CNN filters suffer from a **restricted receptive field**:
* A continuous longitudinal crack extending across 500 pixels cannot be coherently linked by local receptive fields without excessive pooling (which destroys spatial resolution).
* Standard $C2f$ blocks introduce parameter and FLOP redundancy in deep feature maps.

---

### 2.2 Component 1: BOT Module (Bottleneck Transformer)

#### Architectural Role:
Positioned at the end of the backbone (Stage 4 / P5 scale, spatial resolution downsampled to $\frac{H}{32} \times \frac{W}{32}$ with 256–1024 channels).

#### Working Mechanism:
Replaces standard spatial convolutional operations with **Multi-Head Self-Attention (MHSA)**. While convolutions apply static, data-agnostic local weights, MHSA calculates dynamic, pairwise pixel-to-pixel similarity across the entire feature plane, enabling the network to learn global structural relationships across long cracks.

```
Input Feature Map X [C × H × W]
       │
       ├─────────────────────────────────────────┐ (Residual Shortcut)
       ▼                                         │
 1×1 Conv (Channel Reduction: C → d)             │
       ▼                                         │
 Multi-Head Self-Attention (MHSA, h=4 heads)      │
       ▼                                         │
 BatchNorm2d                                     │
       ▼                                         │
 1×1 Conv (Channel Restoration: d → C)           │
       ▼                                         │
 Element-wise Addition (+) ◄─────────────────────┘
       ▼
 SiLU Non-linear Activation
```

#### Mathematical Formulation:
Given an input feature tensor $X \in \mathbb{R}^{N \times d}$ where $N = H \times W$ represents spatial positions and $d$ is the embedding dimension:
1. **Linear Projections:** Linear mappings generate Query ($Q$), Key ($K$), and Value ($V$) tensors:
   $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
   where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$.

2. **Scaled Dot-Product Attention:**
   $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   * $\frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{N \times N}$: Affinity matrix measuring how every spatial coordinate correlates with every other coordinate.
   * $\sqrt{d_k}$: Scaling factor preventing gradient saturation in the Softmax operation for large dimensional inner products.

3. **Multi-Head Aggregation:** For $h$ parallel attention heads:
   $$\text{MHSA}(X) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W_O$$
   where $\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$ and $W_O \in \mathbb{R}^{d \times d}$.

---

### 2.3 Component 2: LSKA (Large Separable Kernel Attention)

#### Architectural Role:
Inserted at the P5 output of the backbone and lateral feature stages (P3, P4, P5) in the feature aggregation neck.

#### Working Mechanism:
Large convolutional kernels (e.g., $19 \times 19$ to $23 \times 23$) expand the effective receptive field (ERF) to cover expansive alligator crack networks. However, standard 2D large convolutions incur prohibitive parameter and computational costs ($\mathcal{O}(K^2)$). LSKA decomposes a 2D large kernel into:
1. A local Depthwise Convolution ($5 \times 5$).
2. A Depthwise Dilated Convolution ($7 \times 7$ with dilation $d=3$, yielding an effective receptive field of $19 \times 19$).
3. A $1 \times 1$ Pointwise Convolution to blend channel interactions.
4. Sigmoid gating to yield an attention probability map.

```
Feature Map F
     │
     ├──► Depthwise Conv (5×5, groups=C)
     │          ▼
     │    Depthwise Dilated Conv (7×7, dilation=3, groups=C)  [ERF = 19×19]
     │          ▼
     │    Pointwise Conv (1×1)
     │          ▼
     │    BatchNorm2d + Sigmoid Activation [σ] ──► Attention Weight Map [A]
     │                                                     │
     └──────────────────────── (⊗) ◄───────────────────────┘
                                │
                          Gated Output
```

#### Mathematical Formulation:
Let $F \in \mathbb{R}^{C \times H \times W}$ denote the input feature tensor:
$$\bar{F} = \text{DW-Conv}_{5 \times 5}(F)$$
$$\hat{F} = \text{DW-D-Conv}_{7 \times 7, \, d=3}(\bar{F})$$
$$A = \sigma\left(\text{BatchNorm}\left(\text{PW-Conv}_{1 \times 1}(\hat{F})\right)\right)$$
$$\text{Output} = F \otimes A$$
where $\otimes$ denotes Hadamard (element-wise) multiplication and $\sigma(z) = \frac{1}{1 + e^{-z}}$.

#### Computational Efficiency Proof:
For a 2D large kernel of size $K \times K$:
* **Standard 2D Conv FLOPs:** $\mathcal{O}(H \cdot W \cdot C_{in} \cdot C_{out} \cdot K^2)$
* **LSKA Decomposed FLOPs:** $\mathcal{O}\left(H \cdot W \cdot C \cdot (k_1^2 + k_2^2) + H \cdot W \cdot C \cdot 1^2\right)$
For $K = 19$, $k_1 = 5$, $k_2 = 7$:
$$\frac{\text{Parameters}_{\text{LSKA}}}{\text{Parameters}_{\text{Standard 2D}}} \approx \frac{5^2 + 7^2 + 1}{19^2} = \frac{25 + 49 + 1}{361} = \frac{75}{361} \approx 20.7\%$$
**Result:** LSKA achieves an equivalent $19 \times 19$ receptive field with **~79.3% reduction in parameters**.

---

### 2.4 Component 3: C2fGhost Module

#### Architectural Role:
Replaces standard $C2f$ feature fusion blocks across the top-down (FPN) and bottom-up (PAN) neck.

#### Working Mechanism:
In traditional deep networks, many feature maps are mutually redundant (resembling ghost copies of each other). GhostNet demonstrates that instead of spending costly dense convolutions to compute all feature maps, half can be generated via intrinsic convolutions and the remainder synthesized via ultra-cheap depthwise linear transformations.

```
Input X [c_in × H × W]
       │
       ▼
 Primary Conv (1×1): c_in → c_out / 2  (Intrinsic Features Y')
       │
       ├───► Identity Pathway ────────────────────────┐
       │                                              │
       ▼                                              │
 Cheap Operation (3×3 Depthwise Conv, groups=c_out/2)  │
       │                                              │
       ▼                                              │
 Ghost Features Y''                                   │
       │                                              │
       └──────────────► Concat [Y', Y''] ◄────────────┘
                              │
                        Output Feature Map
```

#### Mathematical Formulation:
To generate $C$ output channels from input $X$:
1. **Primary Convolution:**
   $$Y' = X * f_{1\times 1}, \quad Y' \in \mathbb{R}^{\frac{C}{2} \times H \times W}$$
2. **Cheap Linear Transformation:**
   $$y''_{i, j} = \Phi_{i, j}(y'_i), \quad \forall i = 1, \dots, \frac{C}{2}$$
   where $\Phi_{i, j}$ is a $3 \times 3$ depthwise linear convolution kernel.
3. **Concatenation:**
   $$Y = [Y', Y''] \in \mathbb{R}^{C \times H \times W}$$

#### Theoretical Speedup:
$$\text{Speedup Ratio } r_s = \frac{C_{in} \cdot C \cdot k^2}{\frac{C}{2} \cdot C_{in} \cdot 1^2 + \frac{C}{2} \cdot k^2} \approx \frac{C_{in} \cdot k^2}{\frac{1}{2} C_{in} + \frac{1}{2} k^2} \approx 2\times$$
**Result:** Cuts the computational burden of the neck by approximately **50%**.

---

### 2.5 Component 4: LSCD-Head (Lightweight Shared Convolution Detection Head)
* Replaces the separate, parameter-heavy convolution branches for bounding box regression and classification.
* Utilizes a **Shared Depthwise Separable Convolution** base followed by independent $1 \times 1$ prediction layers, reducing head parameter count while harmonizing feature representations.

---

## 3. IMPROVEMENT 2: Paper 2 Pipeline (Segmentation & Depth Characterization)

> **Paper 2 Reference:** *Enhancing Pothole Detection and Characterization: Integrated Segmentation and Depth Estimation in Road Anomaly Systems (2025)*  
> **Target:** Pixel-level pothole segmentation combined with monocular depth estimation for geometric area, depth, and hazard quantification.

---

### 3.1 Instance Segmentation via Prototype Masks

To delineate precise pothole boundaries without heavy per-instance mask decoders, the architecture adopts the **YOLACT/YOLOv8-seg** prototype mask framework:

1. **ProtoNet (Prototype Generation):**
   Attached to the P3 feature map ($\frac{H}{4} \times \frac{W}{4}$), ProtoNet generates $K = 32$ basis prototype masks:
   $$P \in \mathbb{R}^{K \times \frac{H}{4} \times \frac{W}{4}}$$
   These prototypes act like spatial basis functions encoding geometric boundary primitives.

2. **Mask Coefficients ($C_i$):**
   The prediction head simultaneously predicts a vector of $K = 32$ scalar mixing coefficients for each candidate instance $i$:
   $$C_i = [c_{i, 1}, c_{i, 2}, \dots, c_{i, 32}] \in \mathbb{R}^{1 \times K}$$

3. **Instance Mask Assembly:**
   The instance mask is assembled through a linear combination of prototype masks weighted by the predicted coefficients, followed by Sigmoid gating and bounding box cropping:
   $$M_i(x, y) = \sigma\left(\sum_{k=1}^{K} c_{i, k} \cdot P_k(x, y)\right)$$
   $$\text{Binary Mask: } B_i(x, y) = \mathbb{I}\left(M_i(x, y) \ge 0.5\right) \cap \text{BBox}_i$$

---

### 3.2 Monocular Depth Estimation Engine (MDE)

Paper 2 evaluates two dedicated Encoder-Decoder depth estimation architectures:

#### (A) DenseDepth Architecture (Section 3.2.1):
* **Encoder:** Pre-trained **DenseNet-169** where dense blocks reuse features via concatenation:
  $$x_\ell = H_\ell([x_0, x_1, \dots, x_{\ell-1}])$$
* **Decoder:** 5-stage bilinear upsampling blocks with skip connections from encoder transition layers to retain fine asphalt surface boundaries.
* **Empirical Metric:** Achieved an **$\text{RMSE} = 1.25$** on unseen road test pairs.

#### (B) Custom Encoder-Decoder Architecture (Section 3.2.2):
* A 4-stage lightweight downsampling convolutional encoder paired with a bilinear upsampling decoder, designed for edge devices.
* **Empirical Metric:** Achieved an **$\text{RMSE} = 1.74$**.

#### (C) Depth Objective Loss Formulation:
Depth networks are supervised on RGB-Depth ground-truth pairs via a composite loss function balancing point accuracy, structural continuity, and high-frequency edge sharpness:
$$L_{\text{total}}(y, \hat{y}) = \lambda_1 L_{\text{depth}}(y, \hat{y}) + \lambda_2 L_{\text{SSIM}}(y, \hat{y}) + \lambda_3 L_{\text{edge}}(y, \hat{y})$$
where:
* **Pointwise Depth Loss:**
  $$L_{\text{depth}}(y, \hat{y}) = \frac{1}{N} \sum_{p=1}^N |y_p - \hat{y}_p|$$
* **Structural Similarity (SSIM) Loss:**
  $$L_{\text{SSIM}}(y, \hat{y}) = \frac{1 - \text{SSIM}(y, \hat{y})}{2} = \frac{1}{2} \left(1 - \frac{(2\mu_y \mu_{\hat{y}} + C_1)(2\sigma_{y\hat{y}} + C_2)}{(\mu_y^2 + \mu_{\hat{y}}^2 + C_1)(\sigma_y^2 + \sigma_{\hat{y}}^2 + C_2)}\right)$$
* **Edge / Gradient Loss:**
  $$L_{\text{edge}}(y, \hat{y}) = \frac{1}{N} \sum_{p=1}^N \left( |\nabla_x y_p - \nabla_x \hat{y}_p| + |\nabla_y y_p - \nabla_y \hat{y}_p| \right)$$

---

### 3.3 Mathematical Formulation of Pothole Characterization (Section 5.1 & Pages 20–22)

Once the segmentation mask $B(x, y)$ and normalized monocular depth map $D(x, y) \in [0.0, 1.0]$ are computed, the characterization engine applies the exact formulas established in Paper 2:

#### 1. Pothole Contour Area ($\text{Area}_{\text{pixels}}$)
Extracting the closed contour $\mathcal{C} = \{(x_1, y_1), \dots, (x_n, y_n)\}$ of the binary mask, the area is evaluated using Green's Theorem (Surveyor's Formula):
$$\text{Area}_{\text{pothole}} = \frac{1}{2} \left| \sum_{j=1}^{n-1} (x_j y_{j+1} - x_{j+1} y_j) + (x_n y_1 - x_1 y_n) \right| \quad (\text{pixels})$$

#### 2. Percentage of Road Damage ($\%$)
Given an image resolution of $H \times W$ (Total Image Area $= H \times W$ pixels):
$$\text{Percentage of Damage (\%)} = \left(\frac{\sum_{i=1}^{N} \text{Area}_{\text{pothole}, i}}{H \times W}\right) \times 100$$
*(Benchmark from Paper 2, Page 20: Example 'a' yields Area $= 12,101.0\text{ px}$ over $245,760\text{ px} \implies \mathbf{4.92\%}$ damage)*.

#### 3. Pothole Depth ($PD$)
The mean normalized depth across the interior pothole domain $\Omega_{\text{pothole}} = \{(x, y) \mid B(x, y) = 1\}$:
$$PD = \frac{1}{|\Omega_{\text{pothole}}|} \sum_{(x, y) \in \Omega_{\text{pothole}}} D(x, y)$$

#### 4. Surrounding Road Surface Depth ($SD$)
To eliminate camera angle slant and perspective bias, Paper 2 strictly constrains surrounding road measurement to an immediate boundary ring around the contour. Using an elliptical morphological structuring element $\mathcal{K}$ (radius 21 pixels):
$$\Omega_{\text{dilated}} = \Omega_{\text{pothole}} \oplus \mathcal{K}$$
$$\Omega_{\text{surround}} = \Omega_{\text{dilated}} \setminus \Omega_{\text{pothole}}$$
$$SD = \frac{1}{|\Omega_{\text{surround}}|} \sum_{(x, y) \in \Omega_{\text{surround}}} D(x, y)$$

#### 5. Depth Cavity Validation Condition
$$PD > SD \implies \text{Confirmed Depression! The depth estimation algorithm succeeds in detecting the cavity.}$$

#### 6. Relative Pothole Depth ($RPD$ in \%)
Defined in Paper 2 as the normalized relative depth ratio indicating how severely the cavity indents below the baseline road grade:
$$RPD = \left(\frac{PD - SD}{SD}\right) \times 100$$
Complementary Absolute Depth Difference Percentage:
$$\Delta D = (PD - SD) \times 100$$
*(Benchmark from Paper 2, Page 21: Figure 18a yields $PD = 0.7693, SD = 0.5808 \implies RPD = \mathbf{18.85\%}$)*.

#### 7. Hazard Severity Index
Paper 2 combines the spatial extent ($\text{Area}_{\text{pothole}}$) and depression severity ($RPD$) into a civil road hazard grade:
$$\text{Hazard Grade} = \begin{cases}
\textbf{Severe}, & \text{if } RPD \ge 18.0\% \text{ or } (\text{Area} > 8000\text{ px} \text{ and } RPD \ge 10.0\%) \\
\textbf{High}, & \text{if } RPD \ge 12.0\% \text{ or } (\text{Area} > 4000\text{ px} \text{ and } RPD \ge 6.0\%) \\
\textbf{Moderate}, & \text{if } RPD \ge 5.0\% \\
\textbf{Low}, & \text{otherwise}
\end{cases}$$

---

## 4. OUR UNIFIED HYBRID ARCHITECTURE (YOLOv8-PD-Seg + Depth)

### 4.1 How Both Papers Merge Into One System

```
                         Input RGB Image [3 × 640 × 640]
                                        │
           ┌────────────────────────────┴───────────────────────────┐
           ▼                                                        ▼
┌──────────────────────────────────────┐         ┌──────────────────────────────────────┐
│  BRANCH 1: YOLOv8-PD-Seg (Ours)      │         │  BRANCH 2: Monocular Depth (Paper 2) │
│                                      │         │                                      │
│  [Backbone - Paper 1 Architecture]   │         │  [DenseDepth / MiDaS / EncDec]       │
│  ├── Stem Conv (/4)                  │         │  ├── DenseNet-169 / Conv Encoder     │
│  ├── C2fGhost Stages (P2, P3, P4)    │         │  └── Bilinear Skip Decoder           │
│  ├── BOT Module (P5, Global MHSA)    │         │                                      │
│  └── LSKA Module (19×19 Attention)   │         │                                      │
│                  │                   │         │                                      │
│  [Neck - Paper 1 Architecture]       │         │                                      │
│  ├── FPN Top-Down (C2fGhost)         │         │                                      │
│  ├── PAN Bottom-Up (C2fGhost)        │         │                                      │
│  └── LSKA at scales P3, P4, P5       │         │                                      │
│                  │                   │         │                                      │
│  [Head - Novel LSCD-Seg-Head]        │         │                                      │
│  ├── Proto Module ──► 32 Basis Masks │         │                                      │
│  └── Shared Conv ───► Cls + BBox +   │         │                                      │
│                       Mask Coeffs C_i│         │                                      │
└──────────────────┬───────────────────┘         └──────────────────┬───────────────────┘
                   │                                                │
         Instance Mask M_i(x, y)                          Dense Depth Map D(x, y)
                   │                                                │
                   └────────────────────┬───────────────────────────┘
                                        ▼
                   ┌────────────────────────────────────────┐
                   │    POTHOLE CHARACTERIZATION ENGINE     │
                   │                                        │
                   │  1. Contour Area: Area_pothole (px)    │
                   │  2. Road Damage: Damage %              │
                   │  3. Cavity Check: PD > SD              │
                   │  4. Relative Depth: RPD %              │
                   │  5. Hazard Grade: Severe / High / Mod  │
                   └────────────────────┬───────────────────┘
                                        ▼
                   ┌────────────────────────────────────────┐
                   │  4-Panel Characterization HUD Output   │
                   │  [Orig | SegMask | DepthMap | HUD]     │
                   └────────────────────────────────────────┘
```

---

### 4.2 The Novel LSCD-Seg-Head Design
In our custom `models/modules.py` and `models/yolov8_pd_seg.py`, we redesigned Paper 1's detection head to natively support instance segmentation while preserving shared convolution parameter savings:

```python
class LSCDSegHead(nn.Module):
    """
    Lightweight Shared Convolution Detection + Segmentation Head.
    Outputs:
      1. Class Logits (C classes)
      2. Bounding Box Coordinates (4 * reg_max)
      3. Mask Coefficients (32 scalar coefficients for Proto mixing)
    """
    def __init__(self, in_channels, num_classes, num_protos=32, reg_max=16):
        super().__init__()
        # Shared feature extraction (Paper 1 lightweight design)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_head = nn.Conv2d(in_channels, 4 * reg_max, 1)
        self.mask_head = nn.Conv2d(in_channels, num_protos, 1)  # ⭐ Novel Mask Branch

    def forward(self, x):
        feat = self.shared_conv(x)
        return self.cls_head(feat), self.reg_head(feat), self.mask_head(feat)
```

---

## 5. Architectural & Capability Comparison Matrix

| Architectural Feature | Baseline YOLOv8n | Paper 1 (YOLOv8-PD) | Paper 2 (Characterization) | **Our Unified Model** |
| :--- | :---: | :---: | :---: | :---: |
| **Primary Task** | Bounding Box Detection | Bounding Box Detection | Segmentation + Depth | **Segmentation + Depth** |
| **Backbone Global Attention** | ❌ None | **BOT (Self-Attention)** | ❌ None | **BOT (Self-Attention)** |
| **Large Receptive Field Attention**| ❌ Local 3×3 | **LSKA ($19\times 19$ ERF)** | ❌ None | **LSKA ($19\times 19$ ERF)** |
| **Neck Computational Optimization**| Standard C2f | **C2fGhost ($\approx 50\%$ FLOPs)** | Standard C2f | **C2fGhost ($\approx 50\%$ FLOPs)** |
| **Prediction Head** | Decoupled Conv | **LSCD (Shared Conv)** | Standard Segment Head | **LSCD-Seg Head (Shared+Mask)** |
| **Instance Segmentation** | ❌ No | ❌ No | ✅ Yes (Standard) | **✅ Yes (Improved Arch)** |
| **Monocular Depth Engine** | ❌ No | ❌ No | ✅ DenseDepth / Custom | **✅ DenseDepth / MiDaS / Custom** |
| **Contour Area Calculation** | ❌ No | ❌ No | ✅ Yes | **✅ Yes ($\oint \text{contour}$)** |
| **Road Damage Percentage ($\%$)** | ❌ No | ❌ No | ✅ Yes | **✅ Yes (Table Page 20)** |
| **Cavity Validation ($PD > SD$)** | ❌ No | ❌ No | ✅ Yes | **✅ Yes (Section 5.1)** |
| **Relative Depth ($RPD$)** | ❌ No | ❌ No | ✅ Yes | **✅ Yes ($\frac{PD-SD}{SD} \times 100$)** |
| **Hazard Severity Classification** | ❌ No | ❌ No | ✅ Yes | **✅ Yes (Severe/High/Mod)** |

---

## 6. Execution Commands & File Mapping

### 6.1 Repository Files:
* `models/modules.py`: PyTorch implementations of `BOTModule`, `LSKAModule`, `C2fGhost`, `ProtoModule`, and `LSCDSegHead`.
* `models/yolov8_pd_seg.py`: Full hybrid model architecture builder.
* `configs/yolov8-pd-seg.yaml`: Model configuration file for Ultralytics YOLO engine.
* `models/depth_estimator.py`: Complete implementation of `DenseDepthModel`, `CustomEncoderDecoder`, `DepthEstimator`, and `extract_pothole_depth()`.
* `characterize_potholes.py`: End-to-end execution script producing the 4-panel visual report and terminal summary table.
* `train.py`: Multi-model training pipeline supporting partial dataset fractions and custom epochs.

### 6.2 Running Characterization:
```bash
python characterize_potholes.py \
  --image test_images/pothole_test.png \
  --model runs/segment/custom_yolov8_pd_seg/weights/best.pt \
  --depth midas \
  --output runs/characterization_report.jpg
```

---
*Created as part of the CV_Project repository. Synchronized with GitHub: [https://github.com/Mahendra194/CV_Project](https://github.com/Mahendra194/CV_Project)*
