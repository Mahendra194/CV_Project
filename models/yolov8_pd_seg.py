"""
YOLOv8-PD-Seg: Full Model Definition
=====================================
Combines:
  - Paper 1 improvements: BOT, LSKA, C2fGhost, LSCD-Head
  - Paper 2 improvement: Instance Segmentation (Proto + Mask coefficients)

This file registers all custom modules with Ultralytics so they can be
used in the YAML config, and provides the model builder function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom modules (handle both package and direct execution)
try:
    from models.modules import (
        C2fGhost, BOTModule, LSKAModule, ProtoModule,
        LSCDSegHead, GhostModule, GhostBottleneck, Conv,
    )
except ModuleNotFoundError:
    from modules import (
        C2fGhost, BOTModule, LSKAModule, ProtoModule,
        LSCDSegHead, GhostModule, GhostBottleneck, Conv,
    )

# Try to register with Ultralytics
try:
    from ultralytics.nn.modules import C2f
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel
    import ultralytics.nn.tasks as tasks

    # Register custom modules so YAML config can reference them
    _original_module_map = getattr(tasks, 'MODULE_MAP', {})
    CUSTOM_MODULES = {
        'C2fGhost': C2fGhost,
        'BOTModule': BOTModule,
        'LSKAModule': LSKAModule,
        'ProtoModule': ProtoModule,
        'LSCDSegHead': LSCDSegHead,
        'GhostModule': GhostModule,
    }
except ImportError:
    print("[INFO] Ultralytics not installed yet. Run: pip install ultralytics")


class YOLOv8PDSeg(nn.Module):
    """
    YOLOv8-PD-Seg: Improved Road Damage Detection with Instance Segmentation.

    Architecture Overview:
    =====================
    
    BACKBONE (Feature Extraction):
    ├── Conv layers (stem)
    ├── C2f blocks (standard YOLOv8)
    ├── BOT Module (Paper 1) ← Global context for long cracks
    └── LSKA Module (Paper 1) ← Large kernel attention

    NECK (Feature Fusion):
    ├── FPN (Top-down) with C2fGhost (Paper 1) ← Lightweight
    └── PAN (Bottom-up) with C2fGhost (Paper 1)

    HEAD (Prediction):
    ├── LSCD-Seg-Head (Paper 1 + Paper 2):
    │   ├── Classification branch → damage type
    │   ├── Bbox regression branch → location
    │   └── Mask coefficient branch → 32 coefficients ⭐
    └── Proto Module (Paper 2) → 32 prototype masks ⭐

    OUTPUT:
    ├── Bounding boxes + class labels (same as Paper 1)
    └── Instance segmentation masks (NEW from Paper 2) ⭐
    """

    def __init__(self, num_classes=4, num_protos=32):
        super().__init__()
        self.num_classes = num_classes
        self.num_protos = num_protos

        # =====================================================================
        # BACKBONE — with Paper 1's BOT Module
        # =====================================================================

        # Stem
        self.stem = nn.Sequential(
            Conv(3, 16, 3, 2),     # /2 → 320×320
            Conv(16, 32, 3, 2),    # /4 → 160×160
        )

        # Stage 1 — P2 (160×160)
        self.stage1 = nn.Sequential(
            Conv(32, 64, 3, 2),    # /8 → 80×80
            C2fGhost(64, 64, n=1),
        )

        # Stage 2 — P3 (80×80) - used for Proto Module
        self.stage2 = nn.Sequential(
            Conv(64, 128, 3, 2),   # /16 → 40×40
            C2fGhost(128, 128, n=2),
        )

        # Stage 3 — P4 (40×40)
        self.stage3 = nn.Sequential(
            Conv(128, 256, 3, 2),  # /32 → 20×20
            C2fGhost(256, 256, n=2),
        )

        # Stage 4 — P5 (20×20) + BOT Module (Paper 1)
        self.stage4 = nn.Sequential(
            Conv(256, 256, 3, 2),  # /64 → 10×10
            C2fGhost(256, 256, n=1),
            BOTModule(256, 256, n=1, num_heads=4),  # ← Paper 1: Global context
        )

        # LSKA at end of backbone (Paper 1)
        self.backbone_lska = LSKAModule(256)

        # =====================================================================
        # NECK — FPN + PAN with C2fGhost (Paper 1)
        # =====================================================================

        # Top-down (FPN)
        self.lateral_p5 = Conv(256, 256, 1)
        self.lateral_p4 = Conv(256, 128, 1)
        self.lateral_p3 = Conv(128, 64, 1)

        self.fpn_c2f_p4 = C2fGhost(256 + 128, 128, n=1)   # Reduced: was 384
        self.fpn_c2f_p3 = C2fGhost(128 + 64, 64, n=1)

        # Bottom-up (PAN)
        self.pan_down_p3 = Conv(64, 64, 3, 2)
        self.pan_c2f_p4 = C2fGhost(64 + 128, 128, n=1)
        self.pan_down_p4 = Conv(128, 128, 3, 2)
        self.pan_c2f_p5 = C2fGhost(128 + 256, 256, n=1)

        # LSKA in neck (Paper 1)
        self.neck_lska_p3 = LSKAModule(64)
        self.neck_lska_p4 = LSKAModule(128)
        self.neck_lska_p5 = LSKAModule(256)

        # =====================================================================
        # HEAD — LSCD-Seg-Head (Paper 1 + Paper 2)
        # =====================================================================
        self.head_p3 = LSCDSegHead(64, num_classes, num_protos)
        self.head_p4 = LSCDSegHead(128, num_classes, num_protos)
        self.head_p5 = LSCDSegHead(256, num_classes, num_protos)

        # =====================================================================
        # PROTO MODULE — Prototype Mask Generator (Paper 2)
        # =====================================================================
        self.proto = ProtoModule(64, proto_channels=128, num_protos=num_protos)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.
        
        Input:  x — [B, 3, 640, 640] image
        Output: dict with:
            - 'cls_preds': list of class predictions from 3 scales
            - 'reg_preds': list of bbox predictions from 3 scales
            - 'mask_preds': list of mask coefficient predictions from 3 scales
            - 'proto_out': prototype masks [B, 32, H, W]
        """
        # ---- BACKBONE ----
        x = self.stem(x)           # [B, 32, 160, 160]
        p2 = self.stage1(x)        # [B, 64, 80, 80]
        p3 = self.stage2(p2)       # [B, 128, 40, 40]
        p4 = self.stage3(p3)       # [B, 256, 20, 20]
        p5 = self.stage4(p4)       # [B, 256, 10, 10]
        p5 = self.backbone_lska(p5)  # LSKA attention

        # ---- NECK: Top-down (FPN) ----
        p5_lat = self.lateral_p5(p5)                           # [B, 256, 10, 10]
        p5_up = F.interpolate(p5_lat, size=p4.shape[2:], mode='nearest')
        p4_td = self.fpn_c2f_p4(torch.cat([p5_up, self.lateral_p4(p4)], 1))  # [B, 128, 20, 20]

        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        p3_td = self.fpn_c2f_p3(torch.cat([p4_up, self.lateral_p3(p3)], 1))  # [B, 64, 40, 40]

        # ---- NECK: Bottom-up (PAN) ----
        p3_out = self.neck_lska_p3(p3_td)                       # [B, 64, 40, 40]

        p3_down = self.pan_down_p3(p3_out)
        p4_out = self.pan_c2f_p4(torch.cat([p3_down, p4_td], 1))
        p4_out = self.neck_lska_p4(p4_out)                       # [B, 128, 20, 20]

        p4_down = self.pan_down_p4(p4_out)
        p5_out = self.pan_c2f_p5(torch.cat([p4_down, p5_lat], 1))
        p5_out = self.neck_lska_p5(p5_out)                       # [B, 256, 10, 10]

        # ---- HEAD: Detection + Segmentation ----
        cls3, reg3, mask3 = self.head_p3(p3_out)
        cls4, reg4, mask4 = self.head_p4(p4_out)
        cls5, reg5, mask5 = self.head_p5(p5_out)

        # ---- PROTO MODULE: Generate prototype masks ----
        proto_out = self.proto(p3_out)  # [B, 32, 80, 80]

        return {
            'cls_preds': [cls3, cls4, cls5],
            'reg_preds': [reg3, reg4, reg5],
            'mask_preds': [mask3, mask4, mask5],
            'proto_out': proto_out,
        }


def build_model(num_classes=4, num_protos=32):
    """Build YOLOv8-PD-Seg model."""
    model = YOLOv8PDSeg(num_classes=num_classes, num_protos=num_protos)
    return model


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Quick test
    model = build_model(num_classes=4)
    total, trainable = count_parameters(model)
    print(f"YOLOv8-PD-Seg Model Summary:")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Model size (approx):  {total * 4 / 1024 / 1024:.1f} MB")

    # Test forward pass
    dummy = torch.randn(1, 3, 640, 640)
    out = model(dummy)
    print(f"\nOutput shapes:")
    for i, (c, r, m) in enumerate(zip(out['cls_preds'], out['reg_preds'], out['mask_preds'])):
        print(f"  Scale P{i+3}: cls={c.shape}, reg={r.shape}, mask_coeff={m.shape}")
    print(f"  Proto: {out['proto_out'].shape}")
