"""
YOLOv8-PD-Seg: Custom Architectural Modules
===========================================
Integrates lightweight attention mechanisms (BOT, LSKA, C2fGhost, LSCD-Head)
with prototype-based instance segmentation (ProtoModule, Mask coefficients).

All modules are designed to be plug-and-play with Ultralytics YOLOv8.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 1. GhostModule & C2fGhost (Paper 1 — Neck Optimization)
# =============================================================================
class GhostModule(nn.Module):
    """
    GhostModule generates intrinsic feature maps via standard convolution,
    followed by cheap linear depthwise operations to produce ghost feature maps.
    Reduces computational cost and parameter footprint by approximately 50%.
    
    Reference: Han et al., "GhostNet: More Features from Cheap Operations", CVPR 2020.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, act=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Primary convolution — intrinsic feature extraction
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )

        # Cheap linear operation — ghost feature generation via depthwise conv
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)       # Real features
        x2 = self.cheap_operation(x1)    # Ghost features (cheap!)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """
    GhostBottleneck = 2x GhostModule + optional SE attention + residual connection.
    """
    def __init__(self, in_channels, mid_channels, out_channels, dw_kernel_size=3, stride=1):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.ghost1 = GhostModule(in_channels, mid_channels, kernel_size=1)

        # Depthwise conv (optional, for stride > 1)
        if stride > 1:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, dw_kernel_size, stride,
                          dw_kernel_size // 2, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
            )
        else:
            self.dw_conv = nn.Identity()

        self.ghost2 = GhostModule(mid_channels, out_channels, kernel_size=1, act=False)

        # Shortcut for residual
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, dw_kernel_size, stride,
                          dw_kernel_size // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.ghost1(x)
        out = self.dw_conv(out)
        out = self.ghost2(out)

        if self.use_residual:
            return out + residual
        else:
            return out + self.shortcut(residual)


class C2fGhost(nn.Module):
    """
    C2fGhost: C2f block with GhostBottleneck feature extraction.
    Replaces standard bottlenecks with GhostBottlenecks to reduce computational
    cost by ~50% FLOPs while preserving multi-scale feature representation.
    """
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(out_channels * e)  # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2 * self.c, 1, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.m = nn.ModuleList(
            GhostBottleneck(self.c, self.c, self.c) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# =============================================================================
# 2. BOT Module — Bottleneck Transformer (Paper 1 — Backbone Improvement)
# =============================================================================
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) layer for the BOT module.
    Replaces spatial convolutions in deep backbone stages with self-attention
    to capture global pavement context and long-span continuous road cracks.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)  # [B, heads, head_dim, N]

        attn = (q.transpose(-2, -1) @ k) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)

        out = (v @ attn.transpose(-2, -1))  # [B, heads, head_dim, N]
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        return out


class BOTBlock(nn.Module):
    """
    Bottleneck Transformer Block.
    Structure: Conv1x1 -> MHSA -> Conv1x1 + Residual Connection.
    Fuses local convolutional feature representations with global self-attention.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        hidden_dim = dim // 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads)
        self.bn_attn = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn_attn(self.attn(out))
        out = self.conv2(out)
        out = self.act(out + residual)
        return out


class BOTModule(nn.Module):
    """
    BOT Module: Stack of BOTBlocks for the backbone.
    Replaces standard convolution layers to capture global context.
    """
    def __init__(self, in_channels, out_channels, n=1, num_heads=4):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        ) if in_channels != out_channels else nn.Identity()

        self.blocks = nn.Sequential(
            *[BOTBlock(out_channels, num_heads) for _ in range(n)]
        )

    def forward(self, x):
        x = self.conv_in(x)
        return self.blocks(x)


# =============================================================================
# 3. LSKA — Large Separable Kernel Attention (Paper 1 — Attention Mechanism)
# =============================================================================
class LSKAModule(nn.Module):
    """
    Large Separable Kernel Attention (LSKA).
    Provides an extensive effective receptive field (19x19) essential for continuous
    elongated cracks while maintaining lightweight parameter and FLOP bounds via
    decomposed 1D horizontal and vertical depth-wise convolutions.
    
    Reference: Li et al., "Large Selective Kernel Network for Remote Sensing Object Detection", 2023.
    """
    def __init__(self, dim, c2=None, *args, **kwargs):
        super().__init__()
        # Ignore c2 if provided by Ultralytics parse_model routing

        # Decomposed large kernel attention
        self.dw_conv1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.dw_conv2 = nn.Conv2d(dim, dim, 7, padding=9, groups=dim, dilation=3, bias=False)
        self.pw_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        # Generate attention map using large separable kernels
        attn = self.dw_conv1(x)
        attn = self.dw_conv2(attn)
        attn = self.pw_conv(attn)
        attn = self.bn(attn)
        # Apply attention
        out = x * attn.sigmoid()
        return out


# =============================================================================
# 4. Proto Module (Prototype Mask Generator)
# =============================================================================
class ProtoModule(nn.Module):
    """
    Prototype Mask Generator for Instance Segmentation.
    Synthesizes 32 high-resolution base prototype masks from the P3 feature map.
    Linear combinations weighted by predicted instance mask coefficients form
    pixel-accurate polygonal segmentation masks for arbitrary damage shapes.
    
    Architecture:
      Conv3x3 -> BN -> SiLU -> Upsample(2x) -> Conv3x3 -> BN -> SiLU -> Conv1x1
    
    Input:  P3 feature map [B, C, H, W]
    Output: 32 prototype masks [B, 32, 2H, 2W]
    """
    def __init__(self, in_channels, proto_channels=256, num_protos=32):
        super().__init__()
        self.num_protos = num_protos

        self.proto_net = nn.Sequential(
            # Process features
            nn.Conv2d(in_channels, proto_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(proto_channels),
            nn.SiLU(inplace=True),

            # Upsample for higher resolution masks
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Refine
            nn.Conv2d(proto_channels, proto_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(proto_channels),
            nn.SiLU(inplace=True),

            # Output prototype masks
            nn.Conv2d(proto_channels, num_protos, 1),
        )

    def forward(self, x):
        return self.proto_net(x)


# =============================================================================
# 5. LSCD-Seg-Head (Lightweight Shared Convolution Detection & Segmentation Head)
# =============================================================================
class LSCDSegHead(nn.Module):
    """
    Lightweight Shared Convolution Detection + Segmentation Head.
    Integrates shared separable convolutions with three specialized prediction branches:
      1. Classification Branch: Probability distribution over 4 damage classes
      2. Bounding Box Branch: Axis-aligned box regression [x, y, w, h]
      3. Mask Branch: 32 prototype mask weighting coefficients per detected instance
    """
    def __init__(self, in_channels, num_classes, num_protos=32, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.num_protos = num_protos
        self.reg_max = reg_max

        # Shared feature extraction (Paper 1's lightweight approach)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        # Classification branch
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )

        # Bbox regression branch
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 4 * reg_max, 1),
        )

        # NEW: Mask coefficient branch (from Paper 2 concept)
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_protos, 1),
        )

    def forward(self, x):
        shared = self.shared_conv(x)
        cls_out = self.cls_head(shared)
        reg_out = self.reg_head(shared)
        mask_out = self.mask_head(shared)   # mask coefficients
        return cls_out, reg_out, mask_out


# =============================================================================
# Utility: Autopad for convolutions
# =============================================================================
def autopad(k, p=None, d=1):
    """Calculate padding for 'same' convolution."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    """Standard Conv2d + BN + SiLU block (YOLOv8 standard)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
