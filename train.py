"""
Training Script for YOLOv8-PD-Seg
===================================
Trains both baseline (YOLOv8n-seg) and improved (YOLOv8-PD-Seg) models
so you can compare results.

Usage:
  # Train improved model (YOLOv8-PD-Seg)
  python train.py --model improved

  # Train baseline for comparison
  python train.py --model baseline

  # Train both for full comparison
  python train.py --model both
  
  # Quick test (2 epochs)
  python train.py --model improved --epochs 2
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_baseline(data_yaml, epochs=100, imgsz=640, batch=16, device='', project_dir='runs', fraction=1.0):
    """
    Train baseline YOLOv8n-seg (vanilla, no improvements).
    This is what Paper 1 would look like if it used segmentation but no improvements.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("  TRAINING: Baseline YOLOv8n-seg (no improvements)")
    print("=" * 70)

    model = YOLO('yolov8n-seg.pt')  # Pretrained YOLOv8n-seg

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name='baseline_yolov8n_seg',
        fraction=fraction,     # NEW: fraction of dataset
        patience=20,           # Early stopping
        save=True,
        save_period=10,        # Save checkpoint every 10 epochs
        plots=True,            # Generate training plots
        val=True,
        verbose=True,
        # Hyperparameters
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    print(f"\n[BASELINE] Training complete!")
    print(f"[BASELINE] Best model: {project_dir}/baseline_yolov8n_seg/weights/best.pt")
    return results


def train_improved(data_yaml, epochs=100, imgsz=640, batch=16, device='', project_dir='runs', fraction=1.0):
    """
    Train improved YOLOv8-PD-Seg model.
    
    Approach: We use YOLOv8n-seg as the starting point and apply our improvements
    through Ultralytics' built-in customization. The custom modules (C2fGhost, BOT, LSKA)
    enhance the architecture while maintaining compatibility with the training pipeline.
    
    Key improvements applied:
    1. BOT Module (Paper 1) — captured via custom model config
    2. LSKA Mechanism (Paper 1) — attention in backbone/neck
    3. C2fGhost (Paper 1) — lightweight neck
    4. Instance Segmentation (Paper 2) — YOLOv8-seg base
    """
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("  TRAINING: Improved YOLOv8-PD-Seg")
    print("  (Paper 1: BOT + LSKA + C2fGhost + LSCD | Paper 2: Segmentation)")
    print("=" * 70)

    # 1. Register custom modules
    from register_modules import register_custom_modules
    register_custom_modules()

    # 2. Build model from YAML
    # We use our custom YAML config to build the architecture
    model = YOLO('configs/yolov8-pd-seg.yaml')
    
    # 3. Load pretrained weights for matching layers (optional but recommended)
    # This transfers weights for standard layers (Conv, Segment) and randomizes custom modules
    model.load('yolov8n-seg.pt')
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name='improved_yolov8_pd_seg',
        fraction=fraction,     # NEW: fraction of dataset
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        val=True,
        verbose=True,
        # Hyperparameters (tuned for road damage)
        optimizer='AdamW',
        lr0=0.0008,            # Slightly lower LR for stability
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,       # Longer warmup for transformer components
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Enhanced augmentation for road damage
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,           # Slight rotation (roads aren't always straight)
        translate=0.15,
        scale=0.5,
        shear=2.0,             # Slight perspective changes
        fliplr=0.5,
        flipud=0.0,            # No vertical flip (roads are always horizontal-ish)
        mosaic=1.0,
        mixup=0.15,            # Slightly more mixup for robustness
        copy_paste=0.1,        # Copy-paste augmentation for small damages
    )

    print(f"\n[IMPROVED] Training complete!")
    print(f"[IMPROVED] Best model: {project_dir}/improved_yolov8_pd_seg/weights/best.pt")
    return results


def train_custom_architecture(data_yaml, epochs=100, imgsz=640, batch=16, device='', project_dir='runs', fraction=1.0):
    """
    Train using our fully custom architecture defined in yolov8_pd_seg.py.
    
    This uses the custom model with all Paper 1 + Paper 2 modules directly,
    integrated with Ultralytics' training pipeline.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("  TRAINING: Custom YOLOv8-PD-Seg Architecture")
    print("  (Full custom: BOT + LSKA + C2fGhost + Proto + LSCD-Seg-Head)")
    print("=" * 70)

    # Load from our custom config YAML
    config_path = str(PROJECT_ROOT / 'configs' / 'yolov8-pd-seg.yaml')

    model = YOLO(config_path)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name='custom_yolov8_pd_seg',
        fraction=fraction,     # NEW: fraction of dataset
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        val=True,
        verbose=True,
        pretrained=False,  # Training from scratch with custom arch
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
    )

    print(f"\n[CUSTOM] Training complete!")
    print(f"[CUSTOM] Best model: {project_dir}/custom_yolov8_pd_seg/weights/best.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8-PD-Seg models')
    parser.add_argument('--model', type=str, default='improved',
                        choices=['baseline', 'improved', 'custom', 'both', 'all'],
                        help='Which model to train')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to dataset.yaml')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=str, default='',
                        help='Device: "" for auto, "cpu", "0", "0,1"')
    parser.add_argument('--project', type=str, default='runs',
                        help='Project directory for saving results')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Fraction of dataset to use (e.g., 0.1 for 10%)')
    args = parser.parse_args()

    # Find dataset yaml
    if args.data:
        data_yaml = args.data
    else:
        # Try common locations
        candidates = [
            PROJECT_ROOT / 'datasets' / 'rdd2022_seg' / 'dataset.yaml',
            PROJECT_ROOT / 'datasets' / 'dataset.yaml',
            PROJECT_ROOT / 'dataset.yaml',
        ]
        data_yaml = None
        for c in candidates:
            if c.exists():
                data_yaml = str(c)
                break
        
        if data_yaml is None:
            print("[ERROR] Dataset not found!")
            print("[INFO] Run 'python prepare_dataset.py' first to download and prepare the dataset.")
            print("[INFO] Or specify path with --data /path/to/dataset.yaml")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  YOLOv8-PD-Seg Training")
    print(f"  Date:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch}")
    print(f"  ImgSz:   {args.imgsz}")
    print(f"  Device:  {args.device or 'auto'}")
    print(f"  Fraction: {args.fraction}")
    print(f"{'='*70}")

    if args.model in ['baseline', 'both', 'all']:
        train_baseline(data_yaml, args.epochs, args.imgsz, args.batch,
                       args.device, args.project, args.fraction)

    if args.model in ['improved', 'both', 'all']:
        train_improved(data_yaml, args.epochs, args.imgsz, args.batch,
                       args.device, args.project, args.fraction)

    if args.model in ['custom', 'all']:
        train_custom_architecture(data_yaml, args.epochs, args.imgsz, args.batch,
                                  args.device, args.project, args.fraction)

    print(f"\n{'='*70}")
    print(f"  All training complete!")
    print(f"  Results saved in: {args.project}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
