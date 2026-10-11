"""
End-to-End Pothole Characterization Pipeline
=============================================
Faithfully implementing the characterization framework from:
"Enhancing Pothole Detection and Characterization:
 Integrated Segmentation and Depth Estimation in Road Anomaly Systems" (Paper 2, 2025)

Workflow:
  1. Base Architecture (Paper 1):
     YOLOv8-PD-Seg (BOT Module, LSKA, C2fGhost, LSCD-Seg-Head) detects and segments road damage.
  2. Depth Estimation (Paper 2):
     DenseDepth / Monocular Depth model predicts per-pixel road elevation/depth.
  3. Integrated Characterization (Paper 2, Section 5.1 & Figures 17-18):
     - Pothole contour area computation (pixels)
     - Total damaged area & Percentage of damage (%)
     - Pothole Depth (PD) vs Surrounding Depth (SD)
     - Relative Pothole Depth (RPD %)
     - Pothole Danger / Severity Grading
     - Comprehensive Multi-Panel Visualization (Figures 16, 17, 18)
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import sys

# Ensure custom modules are available
try:
    from register_modules import register_custom_modules
    from models.depth_estimator import DepthEstimator
except ImportError:
    print("[ERROR] Please run this script from the CV_Project repository root.")
    sys.exit(1)


def draw_hud_box(img, x, y, w, h, bg_color=(20, 20, 20), alpha=0.75):
    """Draw a semi-transparent HUD background box."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)


def characterize_image(image_path, model_path, output_path=None, depth_backend="midas", conf=0.25):
    """
    Executes full Paper 2 characterization pipeline on an input road image.
    """
    print("=" * 75)
    print("  ROAD ANOMALY CHARACTERIZATION SYSTEM (Paper 1 + Paper 2)")
    print("=" * 75)
    print(f"• Input Image:    {image_path}")
    print(f"• Seg Model:      {model_path}")
    print(f"• Depth Backend:  {depth_backend}")
    print("-" * 75)

    # 1. Register custom architecture modules (Paper 1)
    register_custom_modules()

    # 2. Load YOLOv8-PD-Seg Segmentation Model
    print("⏳ Loading YOLOv8-PD-Seg model...")
    seg_model = YOLO(model_path)

    # 3. Load Depth Estimator (Paper 2)
    print("⏳ Initializing Monocular Depth Estimator...")
    depth_estimator = DepthEstimator(model_type=depth_backend)
    depth_estimator.load_model()

    # 4. Read Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image at {image_path}")
        return None

    h, w = img.shape[:2]
    total_image_pixels = h * w
    original_img = img.copy()

    # 5. Run Instance Segmentation (Paper 2 Section 4)
    print(f"⏳ Running segmentation (conf={conf})...")
    results = seg_model(img, conf=conf, verbose=False)
    result = results[0]

    # 6. Generate Monocular Depth Map (Paper 2 Section 3.2)
    print("⏳ Generating dense depth map...")
    depth_map = depth_estimator.estimate_depth(original_img)

    # Prepare visualizations
    seg_mask_only = np.zeros_like(original_img)
    merged_overlay = original_img.copy()

    # Depth colormap
    depth_vis = (depth_map * 255.0).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Class color map (BGR)
    class_colors = {
        0: (255, 120, 0),    # D00 Longitudinal Crack (Blue-ish)
        1: (0, 220, 255),    # D10 Transverse Crack (Yellow)
        2: (255, 0, 200),    # D20 Alligator Crack (Magenta)
        3: (0, 30, 255)      # D40 Pothole (Red)
    }

    pothole_records = []
    other_damage_count = 0
    total_pothole_area = 0.0

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()

        for idx, (m, b) in enumerate(zip(masks, boxes)):
            class_id = int(b[5])
            confidence = float(b[4])
            class_name = result.names.get(class_id, f"Class_{class_id}")
            color = class_colors.get(class_id, (0, 255, 0))

            # Resize mask to original image dimensions
            resized_mask = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (resized_mask > 0.5).astype(np.uint8)

            if binary_mask.sum() == 0:
                continue

            # Draw on pure segmentation mask canvas
            seg_mask_only[binary_mask > 0] = color

            # Blend mask with overlay image
            colored_layer = np.zeros_like(original_img)
            colored_layer[binary_mask > 0] = color
            cv2.addWeighted(colored_layer, 0.45, merged_overlay, 1.0, 0, merged_overlay)

            # Draw boundary contour
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(merged_overlay, contours, -1, color, 2)

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, b[:4])
            cv2.rectangle(merged_overlay, (x1, y1), (x2, y2), color, 1)

            # Check if this anomaly is a pothole (Class D40 in RDD2022 or name contains pothole)
            is_pothole = (class_id == 3) or ("pothole" in class_name.lower())

            if is_pothole:
                # Perform Paper 2 Characterization
                stats = depth_estimator.extract_pothole_depth(depth_map, binary_mask, total_image_pixels)
                stats['id'] = len(pothole_records) + 1
                stats['confidence'] = confidence
                stats['bbox'] = (x1, y1, x2, y2)
                pothole_records.append(stats)
                total_pothole_area += stats['area_pixels']

                # Label on image (Figure 18 style)
                label_txt = f"Pothole #{stats['id']}: Area={int(stats['area_pixels'])}px RPD={stats['RPD']}%"
                cv2.putText(merged_overlay, label_txt, (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
                cv2.putText(merged_overlay, label_txt, (x1, max(22, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
            else:
                other_damage_count += 1
                lbl = f"{class_name} {confidence:.2f}"
                cv2.putText(merged_overlay, lbl, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(merged_overlay, lbl, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    total_damage_percentage = (total_pothole_area / total_image_pixels) * 100.0 if total_image_pixels > 0 else 0.0

    # =========================================================================
    # Terminal Results Table (Matching Paper 2, Section 5.1 & Table on Page 20)
    # =========================================================================
    print("\n" + "=" * 75)
    print("  📊 CHARACTERIZATION SUMMARY (Paper 2, Table Page 20 & 22)")
    print("=" * 75)
    print(f"• Total Image Resolution:   {w} x {h} ({total_image_pixels:,} pixels)")
    print(f"• Total Potholes Detected:  {len(pothole_records)}")
    print(f"• Total Damaged Area:       {total_pothole_area:,.1f} pixels")
    print(f"• Percentage of Damage:     {total_damage_percentage:.2f} %")
    print("-" * 75)

    if pothole_records:
        header = f"{'ID':<4} | {'Area (px)':<10} | {'PD (norm)':<10} | {'SD (norm)':<10} | {'RPD (%)':<9} | {'Cavity?':<8} | {'Danger':<9}"
        print(header)
        print("-" * len(header))
        for p in pothole_records:
            cavity_str = "YES (PD>SD)" if p['cavity_detected'] else "NO"
            print(f"#{p['id']:<3} | {p['area_pixels']:<10.1f} | {p['PD']:<10.4f} | {p['SD']:<10.4f} | {p['RPD']:<9.2f} | {cavity_str:<8} | {p['danger_level']:<9}")
        print("-" * len(header))
    else:
        print("[INFO] No potholes detected in this frame.")

    # =========================================================================
    # Construct 4-Panel Visualization (Paper 2 Figure 17 & 18)
    # =========================================================================
    banner_height = 80
    hud_overlay = merged_overlay.copy()

    # Draw bottom HUD banner with metrics
    draw_hud_box(hud_overlay, 10, h - banner_height - 10, w - 20, banner_height, bg_color=(15, 15, 15), alpha=0.85)
    hud_text_1 = f"Potholes: {len(pothole_records)} | Damaged Area: {total_pothole_area:,.0f} px ({total_damage_percentage:.2f}%)"
    if pothole_records:
        worst_pothole = max(pothole_records, key=lambda x: x['RPD'])
        hud_text_2 = f"Max RPD: {worst_pothole['RPD']:.1f}% (PD={worst_pothole['PD']:.3f}, SD={worst_pothole['SD']:.3f}) | Hazard: {worst_pothole['danger_level'].upper()}"
    else:
        hud_text_2 = "Road surface clear of pothole cavities."

    cv2.putText(hud_overlay, hud_text_1, (25, h - banner_height + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(hud_overlay, hud_text_2, (25, h - banner_height + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 215, 255), 2)

    # 4-Panel Grid layout
    # [ (1) Original Image       ] [ (2) Segmentation Masks   ]
    # [ (3) Monocular Depth Map   ] [ (4) Integrated Overlay   ]
    panel1 = original_img.copy()
    cv2.putText(panel1, "(A) Original Image", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

    panel2 = seg_mask_only.copy()
    cv2.putText(panel2, "(B) Segmented Masks (Paper 2)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    panel3 = depth_colormap.copy()
    cv2.putText(panel3, "(C) Depth Map (Paper 2 DenseDepth)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    panel4 = hud_overlay.copy()
    cv2.putText(panel4, "(D) Characterization HUD (RPD + Area)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    top_row = np.hstack((panel1, panel2))
    bottom_row = np.hstack((panel3, panel4))
    four_panel_grid = np.vstack((top_row, bottom_row))

    # Save output
    if output_path is None:
        stem = Path(image_path).stem
        output_path = f"runs/characterization_{stem}.jpg"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, four_panel_grid)
    print(f"\n✅ 4-Panel characterization visualization saved to: {output_path}")

    return {
        'total_potholes': len(pothole_records),
        'total_pothole_area': total_pothole_area,
        'damage_percentage': total_damage_percentage,
        'records': pothole_records,
        'output_path': output_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pothole Characterization (YOLOv8-PD-Seg + Depth)")
    parser.add_argument("--image", type=str, required=True, help="Path to input road image")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Segmentation model weights")
    parser.add_argument("--depth", type=str, default="midas", choices=["midas", "densedepth", "custom"], help="Depth estimation backend")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Output path for 4-panel image")

    args = parser.parse_args()
    characterize_image(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        depth_backend=args.depth,
        conf=args.conf
    )
