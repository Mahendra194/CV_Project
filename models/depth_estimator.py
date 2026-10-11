"""
Depth Estimation Module (Paper 2 Approach)
============================================
Paper 2 used DenseDepth (encoder-decoder) for monocular depth estimation.
Since we don't have Kinect RGB-Depth pairs for RDD2022, we use Intel's
MiDaS model (pre-trained monocular depth estimator) as our depth backbone.

This follows Paper 2's pipeline:
  1. Take an RGB image
  2. Generate a depth map (relative depth per pixel)
  3. Later, the segmentation mask is merged with this depth map
     to extract pothole depth information.

Reference:
  Paper 2: "Enhancing Pothole Detection and Characterization:
            Integrated Segmentation and Depth Estimation" (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path


class LightweightDepthEstimator(nn.Module):
    """
    Lightweight Encoder-Decoder for Monocular Depth Estimation.
    
    Inspired by Paper 2's custom encoder-decoder architecture.
    Uses a simple encoder (downsampling) + decoder (upsampling) structure
    to predict relative depth from a single RGB image.
    
    This is a lightweight version suitable for edge devices, following
    Paper 2's focus on deployment feasibility.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder (Downsampling path)
        self.enc1 = self._conv_block(3, 32, stride=2)      # /2
        self.enc2 = self._conv_block(32, 64, stride=2)      # /4
        self.enc3 = self._conv_block(64, 128, stride=2)     # /8
        self.enc4 = self._conv_block(128, 256, stride=2)    # /16
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 256)
        
        # Decoder (Upsampling path with skip connections)
        self.dec4 = self._upconv_block(256 + 256, 128)    # /8
        self.dec3 = self._upconv_block(128 + 128, 64)     # /4
        self.dec2 = self._upconv_block(64 + 64, 32)       # /2
        self.dec1 = self._upconv_block(32 + 32, 16)       # /1
        
        # Final depth prediction (1 channel — relative depth)
        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output: 0-1 normalized depth
        )
    
    def _conv_block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)     # [B, 32, H/2, W/2]
        e2 = self.enc2(e1)    # [B, 64, H/4, W/4]
        e3 = self.enc3(e2)    # [B, 128, H/8, W/8]
        e4 = self.enc4(e3)    # [B, 256, H/16, W/16]
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d4 = F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        
        d3 = F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        
        d2 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        d1 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, self._pad_to_match(x[:, :32] if x.shape[1] >= 32 else F.pad(x, (0,0,0,0,0,32-x.shape[1])), d1)], dim=1))
        
        # Due to channel mismatch with input, we handle skip differently for dec1
        depth = self.depth_head(d1)
        
        return depth
    
    def _pad_to_match(self, x, target):
        """Ensure x spatial dims match target."""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x


class DepthEstimator:
    """
    High-level Depth Estimation wrapper.
    
    Uses MiDaS (Intel's pre-trained model) for robust monocular depth estimation.
    This follows Paper 2's approach of using a pre-trained depth model,
    but with MiDaS instead of DenseDepth since:
      1. MiDaS is publicly available and well-tested
      2. It works on any RGB image without needing Kinect depth pairs
      3. It provides relative depth which is sufficient for pothole characterization
    
    Paper 2 trained DenseDepth on their own Kinect RGB-Depth pairs.
    We use MiDaS as it achieves similar results without custom depth data.
    """
    
    def __init__(self, model_type="MiDaS_small"):
        """
        Initialize the depth estimator.
        
        Args:
            model_type: MiDaS model variant.
                - "MiDaS_small" — Fast, lightweight (recommended)
                - "DPT_Hybrid"  — Better quality, slower
                - "DPT_Large"   — Best quality, slowest
        """
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self):
        """Load the MiDaS depth estimation model."""
        print(f"[DepthEstimator] Loading MiDaS ({self.model_type})...")
        
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model.to(self.device)
            self.model.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            print(f"[DepthEstimator] MiDaS loaded successfully ✅")
            return True
            
        except Exception as e:
            print(f"[DepthEstimator] MiDaS load failed: {e}")
            print("[DepthEstimator] Falling back to simple gradient-based depth estimation")
            self.model = None
            return False
    
    def estimate_depth(self, image):
        """
        Estimate depth from an RGB image.
        
        Args:
            image: numpy array (H, W, 3) BGR format (from cv2.imread)
        
        Returns:
            depth_map: numpy array (H, W) with relative depth values (0-1 normalized)
                       Higher values = farther from camera
        """
        h, w = image.shape[:2]
        
        if self.model is not None:
            # Use MiDaS
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            
            # Normalize to 0-1
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            
            return depth_map.astype(np.float32)
        
        else:
            # Fallback: Simple gradient-based pseudo-depth
            # (Not as good as MiDaS, but works without downloading model)
            return self._fallback_depth(image)
    
    def _fallback_depth(self, image):
        """
        Simple gradient-based depth estimation as fallback.
        Uses image gradients and vertical position as depth proxy.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        
        # Laplacian (texture/edges indicate closer objects)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        texture = np.abs(laplacian)
        texture = cv2.GaussianBlur(texture, (31, 31), 0)
        
        # Normalize
        if texture.max() > 0:
            texture = texture / texture.max()
        
        # Vertical gradient (things lower in image are closer — road perspective)
        vertical = np.linspace(0.3, 1.0, h).reshape(-1, 1)
        vertical = np.tile(vertical, (1, w))
        
        # Combine
        depth = 0.6 * vertical + 0.4 * (1 - texture)
        depth = np.clip(depth, 0, 1)
        
        return depth.astype(np.float32)
    
    def extract_pothole_depth(self, depth_map, mask):
        """
        Extract depth information for a specific pothole region.
        (Paper 2, Section: Merging segmentation with depth map)
        
        Args:
            depth_map: Full image depth map (H, W), 0-1 normalized
            mask: Binary mask of the pothole region (H, W), 0 or 1
        
        Returns:
            dict with depth statistics for the pothole region
        """
        # Ensure same size
        if mask.shape != depth_map.shape:
            mask = cv2.resize(mask.astype(np.float32), 
                            (depth_map.shape[1], depth_map.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)
        
        # Extract depth values within the pothole mask
        pothole_depths = depth_map[mask > 0]
        
        if len(pothole_depths) == 0:
            return {
                'mean_depth': 0.0,
                'max_depth': 0.0,
                'min_depth': 0.0,
                'std_depth': 0.0,
                'area_pixels': 0,
                'relative_depth': 0.0,
            }
        
        # Calculate surrounding road depth (context)
        # Dilate the mask to get surrounding area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        dilated = cv2.dilate(mask, kernel)
        surrounding = dilated - mask
        surrounding_depths = depth_map[surrounding > 0]
        
        road_depth = np.median(surrounding_depths) if len(surrounding_depths) > 0 else np.median(pothole_depths)
        
        # Relative depth (how much deeper is the pothole compared to road surface)
        relative_depth = abs(np.mean(pothole_depths) - road_depth)
        
        return {
            'mean_depth': float(np.mean(pothole_depths)),
            'max_depth': float(np.max(pothole_depths)),
            'min_depth': float(np.min(pothole_depths)),
            'std_depth': float(np.std(pothole_depths)),
            'area_pixels': int(mask.sum()),
            'road_surface_depth': float(road_depth),
            'relative_depth': float(relative_depth),
        }
