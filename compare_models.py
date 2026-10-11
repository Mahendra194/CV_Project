"""
Model Comparison Script: Baseline vs Improved YOLOv8-PD-Seg
=============================================================
Runs both models SEPARATELY on the same road image and produces:
  1. Baseline-only output (bounding boxes + masks)
  2. Improved-only output (bounding boxes + masks)
  3. Side-by-side comparison image

Usage:
  # Compare using the test pothole image
  python compare_models.py --image test_images/pothole_test.png

  # Compare with a specific image
  python compare_models.py --image path/to/road_image.jpg

  # Run ONLY the baseline model
  python compare_models.py --image test_images/pothole_test.png --model baseline

  # Run ONLY the improved (custom) model
  python compare_models.py --image test_images/pothole_test.png --model improved

  # Change confidence threshold
  python compare_models.py --image test_images/pothole_test.png --conf 0.15
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Class info ───────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: 'Longitudinal Crack',
    1: 'Transverse Crack',
    2: 'Alligator Crack',
    3: 'Pothole',
}

CLASS_COLORS = {
    0: (80, 120, 255),     # Longitudinal Crack — Coral Red
    1: (100, 230, 130),    # Transverse Crack — Green
    2: (255, 160, 80),     # Alligator Crack — Blue-ish
    3: (60, 220, 255),     # Pothole — Yellow-Orange
}


def load_baseline_model():
    """Load the standard YOLOv8n-seg model (baseline)."""
    from ultralytics import YOLO

    # Priority: trained baseline > pretrained
    candidates = [
        PROJECT_ROOT / 'runs' / 'baseline_yolov8n_seg' / 'weights' / 'best.pt',
        PROJECT_ROOT / 'yolov8n-seg.pt',
    ]
    for p in candidates:
        if p.exists():
            print(f"[BASELINE] Loading: {p}")
            return YOLO(str(p))

    # Fallback: download from ultralytics hub
    print("[BASELINE] Downloading yolov8n-seg.pt ...")
    return YOLO('yolov8n-seg.pt')


def load_improved_model():
    """Load our custom YOLOv8-PD-Seg model (improved)."""
    # Register custom modules FIRST
    import register_modules
    register_modules.register_custom_modules()

    from ultralytics import YOLO

    # Priority: best > last
    candidates = [
        PROJECT_ROOT / 'runs' / 'improved_yolov8_pd_seg' / 'weights' / 'best.pt',
        PROJECT_ROOT / 'runs' / 'segment' / 'runs' / 'test_pipeline' / 'weights' / 'best.pt',
        PROJECT_ROOT / 'runs' / 'segment' / 'runs' / 'test_pipeline' / 'weights' / 'last.pt',
    ]
    for p in candidates:
        if p.exists():
            print(f"[IMPROVED] Loading: {p}")
            return YOLO(str(p))

    print("[ERROR] No trained improved model found!")
    print("  Train first on Kaggle, then copy best.pt here.")
    print("  Expected locations:")
    for p in candidates:
        print(f"    {p}")
    return None


def draw_results_on_image(img, result, model_label, color_border=(0, 200, 255)):
    """
    Draw bounding boxes + segmentation masks on an image.
    Returns the annotated image and a stats dict.
    """
    canvas = img.copy()
    h, w = canvas.shape[:2]

    stats = {
        'detections': 0,
        'has_masks': False,
        'total_mask_area': 0,
        'classes_found': [],
        'avg_confidence': 0.0,
    }

    boxes = result.boxes
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        # No detections — draw a "No detections" label
        cv2.putText(canvas, "No detections", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return canvas, stats

    stats['detections'] = len(boxes)
    confs = []

    # ── Draw masks first (so boxes appear on top) ──
    if masks is not None and len(masks) > 0:
        stats['has_masks'] = True
        for i, mask in enumerate(masks):
            mask_data = mask.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_data, (w, h),
                                       interpolation=cv2.INTER_LINEAR)

            cls_id = int(boxes[i].cls[0]) if i < len(boxes) else 0
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))

            # Create a colored overlay
            overlay = np.zeros_like(canvas, dtype=np.uint8)
            binary = (mask_resized > 0.5).astype(np.uint8)
            overlay[binary == 1] = color

            # Blend overlay onto canvas
            canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.45, 0)

            # Draw mask contour for sharp edges
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 2)

            area = binary.sum()
            stats['total_mask_area'] += area

    # ── Draw bounding boxes + labels ──
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf_val = float(box.conf[0])
        confs.append(conf_val)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        cls_name = CLASS_NAMES.get(cls_id, f'Class_{cls_id}')

        stats['classes_found'].append(cls_name)

        # Draw box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{cls_name} {conf_val:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    stats['avg_confidence'] = np.mean(confs) if confs else 0.0

    # ── Draw model label banner at top ──
    banner_h = 40
    cv2.rectangle(canvas, (0, 0), (w, banner_h), color_border, -1)
    cv2.putText(canvas, model_label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Detection count at bottom
    info = f"Detections: {stats['detections']}"
    if stats['has_masks']:
        info += f" | Mask Area: {stats['total_mask_area']} px"
    cv2.putText(canvas, info, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return canvas, stats


def run_single_model(model_type, image_path, conf=0.25, imgsz=640):
    """Run a single model and save the output."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None

    if model_type == 'baseline':
        model = load_baseline_model()
        label = "BASELINE (YOLOv8n-seg)"
        color = (180, 180, 180)   # Gray banner
    else:
        model = load_improved_model()
        label = "IMPROVED (YOLOv8-PD-Seg)"
        color = (0, 200, 100)     # Green banner

    if model is None:
        return None

    print(f"\n{'='*60}")
    print(f"  Running {label}")
    print(f"  Image: {image_path}")
    print(f"  Conf:  {conf}")
    print(f"{'='*60}")

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device='',
        verbose=False,
    )

    result = results[0]
    annotated, stats = draw_results_on_image(img, result, label, color)

    # Save individual output
    out_dir = PROJECT_ROOT / 'results' / 'comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    out_path = out_dir / f"{stem}_{model_type}.jpg"
    cv2.imwrite(str(out_path), annotated)

    # Print stats
    print(f"\n  Results:")
    print(f"    Detections:     {stats['detections']}")
    print(f"    Has Masks:      {'Yes ✅' if stats['has_masks'] else 'No ❌'}")
    print(f"    Mask Area:      {stats['total_mask_area']} pixels")
    print(f"    Avg Confidence: {stats['avg_confidence']:.2%}")
    print(f"    Classes Found:  {', '.join(stats['classes_found']) or 'None'}")
    print(f"    Saved to:       {out_path}")

    return annotated, stats


def run_comparison(image_path, conf=0.25, imgsz=640):
    """Run both models and create a side-by-side comparison."""

    # ── Step 1: Run Baseline ──
    baseline_result = run_single_model('baseline', image_path, conf, imgsz)

    # ── Step 2: Run Improved ──
    improved_result = run_single_model('improved', image_path, conf, imgsz)

    if baseline_result is None or improved_result is None:
        print("\n[ERROR] Could not run one or both models.")
        return

    baseline_img, baseline_stats = baseline_result
    improved_img, improved_stats = improved_result

    # ── Step 3: Create side-by-side comparison ──
    # Make sure both are the same height
    h1, w1 = baseline_img.shape[:2]
    h2, w2 = improved_img.shape[:2]
    target_h = max(h1, h2)

    if h1 != target_h:
        scale = target_h / h1
        baseline_img = cv2.resize(baseline_img, (int(w1 * scale), target_h))
    if h2 != target_h:
        scale = target_h / h2
        improved_img = cv2.resize(improved_img, (int(w2 * scale), target_h))

    # Add a separator line between the two
    separator = np.ones((target_h, 4, 3), dtype=np.uint8) * 255

    comparison = np.hstack([baseline_img, separator, improved_img])

    # Save comparison
    out_dir = PROJECT_ROOT / 'results' / 'comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    comp_path = out_dir / f"{stem}_comparison.jpg"
    cv2.imwrite(str(comp_path), comparison)

    # ── Summary Table ──
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Baseline':<18} {'Improved':<18}")
    print(f"  {'-'*25} {'-'*18} {'-'*18}")
    print(f"  {'Detections':<25} {baseline_stats['detections']:<18} {improved_stats['detections']:<18}")
    print(f"  {'Has Segmentation':<25} {'Yes' if baseline_stats['has_masks'] else 'No':<18} {'Yes' if improved_stats['has_masks'] else 'No':<18}")
    print(f"  {'Mask Area (px)':<25} {baseline_stats['total_mask_area']:<18} {improved_stats['total_mask_area']:<18}")
    print(f"  {'Avg Confidence':<25} {baseline_stats['avg_confidence']:<18.2%} {improved_stats['avg_confidence']:<18.2%}")
    print(f"  {'Classes Found':<25} {len(set(baseline_stats['classes_found'])):<18} {len(set(improved_stats['classes_found'])):<18}")
    print(f"{'='*60}")
    print(f"\n  Individual outputs:")
    print(f"    Baseline: results/comparison/{stem}_baseline.jpg")
    print(f"    Improved: results/comparison/{stem}_improved.jpg")
    print(f"  Side-by-side:")
    print(f"    {comp_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Compare Baseline vs Improved YOLOv8-PD-Seg on road images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both models and compare
  python compare_models.py --image test_images/pothole_test.png

  # Run only baseline
  python compare_models.py --image test_images/pothole_test.png --model baseline

  # Run only improved (custom)
  python compare_models.py --image test_images/pothole_test.png --model improved

  # Lower confidence for more detections
  python compare_models.py --image test_images/pothole_test.png --conf 0.10
        """,
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the road image to test')
    parser.add_argument('--model', type=str, default='both',
                        choices=['baseline', 'improved', 'both'],
                        help='Which model to run (default: both)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference (default: 640)')

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"  YOLOv8-PD-Seg Model Comparison Tool")
    print(f"  Date:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Image: {image_path}")
    print(f"  Mode:  {args.model}")
    print(f"  Conf:  {args.conf}")
    print(f"{'#'*60}")

    if args.model == 'both':
        run_comparison(str(image_path), args.conf, args.imgsz)
    else:
        run_single_model(args.model, str(image_path), args.conf, args.imgsz)

    print("[DONE] ✅")


if __name__ == '__main__':
    main()
