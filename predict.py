"""
Prediction / Inference Script for YOLOv8-PD-Seg
=================================================
Run inference and visualize results with bounding boxes + segmentation masks.

Usage:
  # Predict on a single image
  python predict.py --model runs/improved_yolov8_pd_seg/weights/best.pt --source image.jpg

  # Predict on a folder of images
  python predict.py --model best.pt --source test_images/

  # Predict with video
  python predict.py --model best.pt --source road_video.mp4

  # Compare baseline vs improved on same images
  python predict.py --compare --source test_images/
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# Class colors for visualization
CLASS_COLORS = {
    0: (255, 100, 100),   # D00 Longitudinal Crack — Red
    1: (100, 255, 100),   # D10 Transverse Crack — Green
    2: (100, 100, 255),   # D20 Alligator Crack — Blue
    3: (255, 255, 100),   # D40 Pothole — Yellow
}

CLASS_NAMES = {
    0: 'Longitudinal Crack',
    1: 'Transverse Crack',
    2: 'Alligator Crack',
    3: 'Pothole',
}


def predict_single(model_path, source, imgsz=640, conf=0.25, device='',
                     save_dir='results/predictions'):
    """Run prediction with a single model."""
    import register_modules
    from ultralytics import YOLO

    model = YOLO(model_path)
    
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=True,
        save_txt=True,
        project=save_dir,
        name=Path(model_path).parent.parent.name,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )

    # Print summary
    for r in results:
        boxes = r.boxes
        masks = r.masks

        print(f"\n[RESULT] Image: {r.path}")
        print(f"  Detections: {len(boxes)}")

        if boxes is not None:
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = CLASS_NAMES.get(cls_id, f'Class_{cls_id}')
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"  [{i+1}] {cls_name}: {conf_val:.2%} at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

                # If mask is available, calculate area
                if masks is not None and i < len(masks):
                    mask = masks[i].data[0].cpu().numpy()
                    area_pixels = mask.sum()
                    print(f"       Mask area: {area_pixels:.0f} pixels")

        if masks is not None:
            print(f"  Segmentation masks: {len(masks)} generated ✅")
        else:
            print(f"  Segmentation masks: Not available (detection-only model)")

    return results


def predict_compare(source, imgsz=640, conf=0.25, device=''):
    """
    Run both baseline and improved models on the same images
    and save side-by-side comparison.
    """
    from ultralytics import YOLO

    runs_dir = PROJECT_ROOT / 'runs'
    baseline_path = runs_dir / 'baseline_yolov8n_seg' / 'weights' / 'best.pt'
    improved_path = runs_dir / 'improved_yolov8_pd_seg' / 'weights' / 'best.pt'

    save_dir = PROJECT_ROOT / 'results' / 'comparison'
    save_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    if baseline_path.exists():
        models['baseline'] = YOLO(str(baseline_path))
    else:
        print(f"[WARNING] Baseline not found: {baseline_path}")

    if improved_path.exists():
        models['improved'] = YOLO(str(improved_path))
    else:
        print(f"[WARNING] Improved not found: {improved_path}")

    if not models:
        print("[ERROR] No trained models found. Train first with: python train.py --model both")
        return

    # Get source images
    source_path = Path(source)
    if source_path.is_file():
        images = [source_path]
    elif source_path.is_dir():
        images = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
    else:
        print(f"[ERROR] Source not found: {source}")
        return

    for img_path in images[:20]:  # Max 20 images
        print(f"\nProcessing: {img_path.name}")

        results_dict = {}
        for name, model in models.items():
            results = model.predict(
                source=str(img_path),
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False,
            )
            results_dict[name] = results[0]

        # Create side-by-side comparison
        create_comparison_image(img_path, results_dict, save_dir)

    print(f"\n[INFO] Comparison images saved to: {save_dir}")


def create_comparison_image(img_path, results_dict, save_dir):
    """Create a side-by-side comparison image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return

    panels = []

    for model_name, result in results_dict.items():
        panel = img.copy()

        # Draw boxes
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))

                cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
                label = f"{CLASS_NAMES.get(cls_id, '?')} {conf_val:.0%}"
                cv2.putText(panel, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw masks (if available)
        if result.masks is not None:
            for i, mask in enumerate(result.masks):
                mask_data = mask.data[0].cpu().numpy()
                # Resize mask to image size
                mask_resized = cv2.resize(mask_data, (panel.shape[1], panel.shape[0]))

                cls_id = int(result.boxes[i].cls[0])
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))

                # Create colored overlay
                overlay = np.zeros_like(panel)
                overlay[mask_resized > 0.5] = color
                panel = cv2.addWeighted(panel, 1.0, overlay, 0.4, 0)

        # Add model name label
        cv2.putText(panel, model_name.upper(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        panels.append(panel)

    # Concatenate side by side
    if len(panels) > 1:
        comparison = np.hstack(panels)
    else:
        comparison = panels[0]

    out_path = save_dir / f"compare_{img_path.stem}.jpg"
    cv2.imwrite(str(out_path), comparison)


def main():
    parser = argparse.ArgumentParser(description='Predict with YOLOv8-PD-Seg')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Image/video/folder path')
    parser.add_argument('--compare', action='store_true',
                        help='Compare baseline vs improved')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()

    if args.compare:
        predict_compare(args.source, args.imgsz, args.conf, args.device)
    elif args.model:
        predict_single(args.model, args.source, args.imgsz, args.conf,
                        args.device)
    else:
        # Try to find trained model automatically
        for candidate in [
            'runs/improved_yolov8_pd_seg/weights/best.pt',
            'runs/baseline_yolov8n_seg/weights/best.pt',
        ]:
            if Path(candidate).exists():
                predict_single(candidate, args.source, args.imgsz, args.conf,
                                args.device)
                return
        print("[ERROR] No model found. Specify --model or train first.")


if __name__ == '__main__':
    main()
