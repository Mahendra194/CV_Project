"""
Depth Estimation & Pothole Characterization Module
===================================================
Faithfully implementing the depth estimation and characterization methodology
from Paper 2:
"Enhancing Pothole Detection and Characterization:
 Integrated Segmentation and Depth Estimation in Road Anomaly Systems" (2025)

Key Components from Paper 2:
  1. Depth Models (Section 3.2):
     - DenseDepth Model (DenseNet-169 Backbone + Bilinear Upsampling Decoder)
     - Custom Encoder-Decoder Depth Estimator
     - Pre-trained Monocular Depth Estimator (MiDaS) for zero-shot road evaluation
  2. Characterization Pipeline (Section 5.1 & Figures 17-18):
     - Pothole Depth (PD): Mean depth inside the segmented pothole contour
     - Surrounding Depth (SD): Mean depth restricted strictly to adjacent surrounding road pixels
     - Depth Cavity Verification: Validates that PD > SD (depression detected)
     - Relative Pothole Depth (RPD): RPD = ((PD - SD) / SD) * 100
     - Percentage of Damage (%): (Total Pothole Area / Total Image Area) * 100
     - Danger Severity Assessment: Combines Area and RPD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path


# =============================================================================
# 1. DenseDepth Architecture (Paper 2, Section 3.2.1)
# =============================================================================
class UpSampleBlock(nn.Module):
    """Upsampling decoder block used in DenseDepth architecture."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, concat_with=None):
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if concat_with is not None:
            if out.shape[2:] != concat_with.shape[2:]:
                out = F.interpolate(out, size=concat_with.shape[2:], mode='bilinear', align_corners=True)
            out = torch.cat([out, concat_with], dim=1)
        out = self.conv_a(out)
        out = self.conv_b(out)
        return out


class DenseDepthModel(nn.Module):
    """
    DenseDepth Model as evaluated in Paper 2 (Section 3.2.1).
    Encoder: DenseNet-169 features
    Decoder: Upsampling blocks with skip connections to extract high-fidelity depth.
    """
    def __init__(self, pretrained_encoder=True):
        super().__init__()
        try:
            import torchvision.models as models
            densenet = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT if pretrained_encoder else None)
            features = densenet.features
            
            # Encoder stages
            self.enc0 = nn.Sequential(features.conv0, features.norm0, features.relu0, features.pool0) # 64 ch, /4
            self.enc1 = features.denseblock1  # 256 ch
            self.trans1 = features.transition1 # 128 ch, /8
            self.enc2 = features.denseblock2  # 512 ch
            self.trans2 = features.transition2 # 256 ch, /16
            self.enc3 = features.denseblock3  # 1280 ch
            self.trans3 = features.transition3 # 640 ch, /32
            self.enc4 = features.denseblock4  # 1664 ch
            
            # Decoder
            self.conv2 = nn.Conv2d(1664, 832, kernel_size=1, stride=1, padding=0)
            self.up1 = UpSampleBlock(832 + 256, 416)     # /16
            self.up2 = UpSampleBlock(416 + 128, 208)     # /8
            self.up3 = UpSampleBlock(208 + 64, 104)      # /4
            self.up4 = UpSampleBlock(104, 52)            # /2
            self.up5 = UpSampleBlock(52, 26)             # /1
            
            self.depth_head = nn.Sequential(
                nn.Conv2d(26, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()  # Normalized depth 0 to 1
            )
            self.available = True
        except Exception as e:
            self.available = False

    def forward(self, x):
        if not self.available:
            raise RuntimeError("Torchvision DenseNet169 unavailable.")
        h, w = x.shape[2:]
        e0 = self.enc0(x)
        e1 = self.trans1(self.enc1(e0))
        e2 = self.trans2(self.enc2(e1))
        e3 = self.trans3(self.enc3(e2))
        e4 = self.enc4(e3)
        
        d0 = self.conv2(e4)
        d1 = self.up1(d0, e2)
        d2 = self.up2(d1, e1)
        d3 = self.up3(d2, e0)
        d4 = self.up4(d3)
        d5 = self.up5(d4)
        out = self.depth_head(d5)
        if out.shape[2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return out


# =============================================================================
# 2. Custom Encoder-Decoder Depth Estimator (Paper 2, Section 3.2.2)
# =============================================================================
class CustomEncoderDecoder(nn.Module):
    """
    Custom lightweight Encoder-Decoder model corresponding to Paper 2's 'Our Model'.
    Trained on RGB-depth pairs for road distress characterization.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(3, 32, stride=2)       # /2
        self.enc2 = self._conv_block(32, 64, stride=2)      # /4
        self.enc3 = self._conv_block(64, 128, stride=2)     # /8
        self.enc4 = self._conv_block(128, 256, stride=2)    # /16
        
        self.bottleneck = self._conv_block(256, 256)
        
        # Decoder with skip connections
        self.dec4 = self._upconv_block(256 + 128, 128)
        self.dec3 = self._upconv_block(128 + 64, 64)
        self.dec2 = self._upconv_block(64 + 32, 32)
        self.dec1 = self._upconv_block(32, 16)
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _conv_block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        
        d4 = F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        
        d3 = F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        
        d2 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        d1 = F.interpolate(d2, size=(h, w), mode='bilinear', align_corners=False)
        d1 = self.dec1(d1)
        
        depth = self.depth_head(d1)
        return depth


# =============================================================================
# 3. High-Level Depth Estimator Controller
# =============================================================================
class DepthEstimator:
    """
    Unified Depth Estimator supporting:
      - 'midas' (Default, pre-trained for reliable zero-shot depth without custom Kinect training)
      - 'densedepth' (Paper 2 Section 3.2.1 architecture)
      - 'custom' (Paper 2 Section 3.2.2 architecture)
    """
    def __init__(self, model_type="midas"):
        self.model_type = model_type.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None

    def load_model(self):
        """Loads selected depth network."""
        print(f"[DepthEstimator] Initializing depth model: {self.model_type} on {self.device}")
        
        if "midas" in self.model_type:
            try:
                variant = "MiDaS_small" if "small" in self.model_type or self.model_type == "midas" else "DPT_Hybrid"
                self.model = torch.hub.load("intel-isl/MiDaS", variant, trust_repo=True)
                self.model.to(self.device).eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                self.transform = midas_transforms.small_transform if variant == "MiDaS_small" else midas_transforms.dpt_transform
                print(f"[DepthEstimator] MiDaS ({variant}) loaded successfully ✅")
                return True
            except Exception as e:
                print(f"[DepthEstimator] Warning: Could not load MiDaS via hub: {e}. Falling back to Custom Encoder-Decoder.")
                self.model_type = "custom"

        if "dense" in self.model_type:
            try:
                self.model = DenseDepthModel(pretrained_encoder=True).to(self.device).eval()
                print("[DepthEstimator] DenseDepth model loaded successfully ✅")
                return True
            except Exception as e:
                print(f"[DepthEstimator] DenseDepth load error: {e}. Falling back.")

        # Custom Encoder-Decoder
        self.model = CustomEncoderDecoder().to(self.device).eval()
        print("[DepthEstimator] Custom Encoder-Decoder model initialized ✅")
        return True

    def estimate_depth(self, image):
        """
        Generates normalized depth map (0.0 to 1.0) from an RGB image.
        In this representation, higher values correspond to greater depth/cavity recess.
        """
        h, w = image.shape[:2]
        
        if self.model is None:
            self.load_model()

        if self.transform is not None:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb).to(self.device)
            with torch.no_grad():
                pred = self.model(input_batch)
                pred = F.interpolate(
                    pred.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False
                ).squeeze()
            depth_map = pred.cpu().numpy()
        else:
            # Direct inference for PyTorch models
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(tensor)
            depth_map = pred.squeeze().cpu().numpy()

        # Normalize depth to [0.0, 1.0]
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)

        return depth_map.astype(np.float32)

    # =========================================================================
    # 4. Pothole Characterization Analysis (Paper 2, Section 5.1 & Fig 17-18)
    # =========================================================================
    def extract_pothole_depth(self, depth_map, binary_mask, total_image_area=None):
        """
        Extracts exact pothole characterization metrics defined in Paper 2:
          - Pothole Depth (PD): Average depth of pothole area
          - Surrounding Depth (SD): Average depth restricted strictly to surrounding boundary pixels
          - Cavity Verification: Validates PD > SD
          - Relative Pothole Depth (RPD %): ((PD - SD) / SD) * 100
          - Absolute Depth Difference (%): (PD - SD) * 100
          - Contour Area: Exact pixel count
          - Road Damage Percentage (%): (Pothole Area / Total Image Area) * 100
          - Danger Level: Severity index based on Area and RPD
        """
        h, w = depth_map.shape[:2]
        if total_image_area is None:
            total_image_area = h * w

        # Resize mask if needed
        if binary_mask.shape != (h, w):
            binary_mask = cv2.resize(binary_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        mask_indices = binary_mask > 0
        pothole_pixels = depth_map[mask_indices]

        if len(pothole_pixels) == 0:
            return {
                'area_pixels': 0.0,
                'percentage_damage': 0.0,
                'PD': 0.0,
                'SD': 0.0,
                'RPD': 0.0,
                'diff_depth_percent': 0.0,
                'cavity_detected': False,
                'danger_level': 'None'
            }

        # 1. Pothole Area via Contours (Paper 2, Page 18 & 20)
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour_area = float(cv2.contourArea(contours[0]))
            # If contour area is zero (tiny fragment), fallback to pixel sum
            area_pixels = contour_area if contour_area > 0 else float(np.sum(mask_indices))
        else:
            area_pixels = float(np.sum(mask_indices))

        # Percentage of damage (Paper 2, Page 20 Table)
        percentage_damage = (area_pixels / total_image_area) * 100.0

        # 2. Pothole Depth (PD) (Paper 2, Page 21)
        PD = float(np.mean(pothole_pixels))

        # 3. Surrounding Depth (SD) (Paper 2, Page 21)
        # Restricted strictly to surrounding ring pixels using morphological dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        dilated_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel)
        surrounding_mask = (dilated_mask > 0) & (~mask_indices)
        surrounding_pixels = depth_map[surrounding_mask]

        if len(surrounding_pixels) > 0:
            SD = float(np.mean(surrounding_pixels))
        else:
            SD = PD

        # 4. Cavity verification (Paper 2, Page 21: "PD > SD indicates detected pothole depth")
        cavity_detected = bool(PD > SD)

        # 5. Relative Pothole Depth (RPD) (Paper 2, Page 21 Equation)
        # RPD = ((PD - SD) / SD) * 100
        if SD > 1e-6:
            RPD = ((PD - SD) / SD) * 100.0
        else:
            RPD = (PD - SD) * 100.0

        diff_depth_percent = (PD - SD) * 100.0

        # 6. Danger Severity Assessment (Paper 2, Page 21)
        # Danger increases with both RPD and area
        if RPD >= 18.0 or (area_pixels > 8000 and RPD >= 10.0):
            danger_level = "Severe"
        elif RPD >= 12.0 or (area_pixels > 4000 and RPD >= 6.0):
            danger_level = "High"
        elif RPD >= 5.0:
            danger_level = "Moderate"
        else:
            danger_level = "Low"

        return {
            'area_pixels': round(area_pixels, 1),
            'percentage_damage': round(percentage_damage, 2),
            'PD': round(PD, 4),
            'SD': round(SD, 4),
            'RPD': round(RPD, 2),
            'diff_depth_percent': round(diff_depth_percent, 2),
            'cavity_detected': cavity_detected,
            'danger_level': danger_level
        }
