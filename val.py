"""
Validation Script for YOLOv8-PD-Seg
====================================
Evaluate trained models and generate comparison metrics.

Usage:
  # Validate improved model
  python val.py --model runs/improved_yolov8_pd_seg/weights/best.pt

  # Compare baseline vs improved
  python val.py --compare

  # Validate with specific dataset
  python val.py --model best.pt --data datasets/rdd2022_seg/dataset.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def validate_model(model_path, data_yaml, imgsz=640, batch=16, device='',
                    conf=0.25, iou=0.7, name='val'):
    """Validate a single model and return metrics."""
    import register_modules
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  Validating: {Path(model_path).parent.parent.name}")
    print(f"{'='*60}")

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=True,
        name=name,
    )

    # Extract key metrics
    results = {
        'model': str(model_path),
        'box_mAP50': float(metrics.box.map50),
        'box_mAP50_95': float(metrics.box.map),
        'box_precision': float(metrics.box.mp),
        'box_recall': float(metrics.box.mr),
    }

    # Segmentation metrics (if available)
    if hasattr(metrics, 'seg'):
        results.update({
            'mask_mAP50': float(metrics.seg.map50),
            'mask_mAP50_95': float(metrics.seg.map),
            'mask_precision': float(metrics.seg.mp),
            'mask_recall': float(metrics.seg.mr),
        })

    return results


def compare_models(data_yaml, imgsz=640, batch=16, device=''):
    """Compare baseline and improved models side by side."""
    runs_dir = PROJECT_ROOT / 'runs'

    baseline_path = runs_dir / 'baseline_yolov8n_seg' / 'weights' / 'best.pt'
    improved_path = runs_dir / 'improved_yolov8_pd_seg' / 'weights' / 'best.pt'

    results = {}

    if baseline_path.exists():
        results['baseline'] = validate_model(
            str(baseline_path), data_yaml, imgsz, batch, device,
            name='val_baseline'
        )
    else:
        print(f"[WARNING] Baseline model not found: {baseline_path}")
        print(f"[INFO] Train it first: python train.py --model baseline")

    if improved_path.exists():
        results['improved'] = validate_model(
            str(improved_path), data_yaml, imgsz, batch, device,
            name='val_improved'
        )
    else:
        print(f"[WARNING] Improved model not found: {improved_path}")
        print(f"[INFO] Train it first: python train.py --model improved")

    # Print comparison table
    if len(results) >= 2:
        print_comparison(results)

    return results


def print_comparison(results):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON RESULTS")
    print(f"{'='*80}")

    header = f"{'Metric':<25} {'Baseline':<15} {'Improved':<15} {'Difference':<15}"
    print(header)
    print("-" * 70)

    baseline = results.get('baseline', {})
    improved = results.get('improved', {})

    metrics_to_compare = [
        ('box_mAP50', 'Box mAP@0.5'),
        ('box_mAP50_95', 'Box mAP@0.5:0.95'),
        ('box_precision', 'Box Precision'),
        ('box_recall', 'Box Recall'),
        ('mask_mAP50', 'Mask mAP@0.5'),
        ('mask_mAP50_95', 'Mask mAP@0.5:0.95'),
        ('mask_precision', 'Mask Precision'),
        ('mask_recall', 'Mask Recall'),
    ]

    for key, display_name in metrics_to_compare:
        b_val = baseline.get(key, None)
        i_val = improved.get(key, None)

        if b_val is not None and i_val is not None:
            diff = i_val - b_val
            sign = '+' if diff >= 0 else ''
            emoji = '✅' if diff > 0 else ('⚠️' if diff == 0 else '❌')
            print(f"  {display_name:<23} {b_val:.4f}        {i_val:.4f}        {sign}{diff:.4f} {emoji}")
        elif i_val is not None:
            print(f"  {display_name:<23} {'N/A':<14} {i_val:.4f}")
        elif b_val is not None:
            print(f"  {display_name:<23} {b_val:.4f}        {'N/A'}")

    print(f"{'='*80}")

    # Save comparison to file
    save_comparison(results)


def save_comparison(results):
    """Save comparison results to a markdown file."""
    output_path = PROJECT_ROOT / 'results' / 'comparison_results.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# YOLOv8-PD-Seg: Model Comparison Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Metrics\n\n")
        f.write("| Metric | Baseline (YOLOv8n-seg) | Improved (YOLOv8-PD-Seg) | Diff |\n")
        f.write("|--------|----------------------|------------------------|------|\n")

        baseline = results.get('baseline', {})
        improved = results.get('improved', {})

        for key in ['box_mAP50', 'box_mAP50_95', 'box_precision', 'box_recall',
                     'mask_mAP50', 'mask_mAP50_95', 'mask_precision', 'mask_recall']:
            b = baseline.get(key, 'N/A')
            i = improved.get(key, 'N/A')
            if isinstance(b, float) and isinstance(i, float):
                diff = f"{i-b:+.4f}"
            else:
                diff = '-'
            b_str = f"{b:.4f}" if isinstance(b, float) else str(b)
            i_str = f"{i:.4f}" if isinstance(i, float) else str(i)
            f.write(f"| {key} | {b_str} | {i_str} | {diff} |\n")

    print(f"\n[INFO] Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate YOLOv8-PD-Seg models')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights (.pt file)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to dataset.yaml')
    parser.add_argument('--compare', action='store_true',
                        help='Compare baseline vs improved')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.7)
    args = parser.parse_args()

    # Find dataset yaml
    if args.data:
        data_yaml = args.data
    else:
        candidates = [
            PROJECT_ROOT / 'datasets' / 'rdd2022_seg' / 'dataset.yaml',
            PROJECT_ROOT / 'datasets' / 'dataset.yaml',
        ]
        data_yaml = None
        for c in candidates:
            if c.exists():
                data_yaml = str(c)
                break
        if data_yaml is None:
            print("[ERROR] Dataset not found. Use --data flag.")
            sys.exit(1)

    if args.compare:
        compare_models(data_yaml, args.imgsz, args.batch, args.device)
    elif args.model:
        results = validate_model(args.model, data_yaml, args.imgsz, args.batch,
                                  args.device, args.conf, args.iou)
        print(f"\n[Results] {results}")
    else:
        print("[ERROR] Specify --model <path> or --compare")
        parser.print_help()


if __name__ == '__main__':
    main()
