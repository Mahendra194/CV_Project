"""
Dataset Preparation for YOLOv8-PD-Seg
======================================
Downloads RDD2022 dataset and converts bounding box annotations
to segmentation-compatible format (polygon masks from bounding boxes).

Since RDD2022 only has bounding box annotations (no segmentation masks),
we generate approximate polygon masks from the bounding boxes.
For production use, proper polygon annotations should be created.

Classes:
  0: D00 — Longitudinal Cracks (lambi darar)
  1: D10 — Transverse Cracks (aadi darar)
  2: D20 — Alligator/Mesh Cracks (jaali waali darar)
  3: D40 — Potholes (gaddhey)
"""

import os
import sys
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import urllib.request
import zipfile


# Class mapping for RDD2022
CLASS_MAP = {
    'D00': 0,  # Longitudinal Crack
    'D10': 1,  # Transverse Crack
    'D20': 2,  # Alligator Crack
    'D40': 3,  # Pothole
}


def download_rdd2022(data_dir):
    """
    Download RDD2022 dataset from the official source.
    Falls back to creating a sample dataset structure if download fails.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    rdd_url = "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Japan.zip"
    zip_path = data_dir / "RDD2022_Japan.zip"
    extract_dir = data_dir / "RDD2022_Japan"
    
    if extract_dir.exists():
        print(f"[INFO] RDD2022 already exists at {extract_dir}")
        return extract_dir
    
    print(f"[INFO] Downloading RDD2022 (Japan) dataset...")
    print(f"[INFO] URL: {rdd_url}")
    print(f"[INFO] This may take a few minutes...")
    
    try:
        urllib.request.urlretrieve(rdd_url, str(zip_path))
        print(f"[INFO] Download complete. Extracting...")
        
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(data_dir))
        
        # Clean up zip
        zip_path.unlink()
        print(f"[INFO] Extraction complete: {extract_dir}")
        return extract_dir
        
    except Exception as e:
        print(f"[WARNING] Auto-download failed: {e}")
        print(f"[INFO] Please download RDD2022 manually:")
        print(f"  1. Go to: https://crddc2022.sekilab.global/data/")
        print(f"  2. Download any country dataset (Japan recommended)")
        print(f"  3. Extract to: {extract_dir}")
        print(f"\n[INFO] Creating sample dataset structure for testing...")
        create_sample_structure(data_dir)
        return data_dir / "sample"


def create_sample_structure(data_dir):
    """Create a minimal sample dataset structure for testing."""
    sample_dir = Path(data_dir) / "sample"
    for split in ['train', 'val']:
        (sample_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (sample_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Sample structure created at {sample_dir}")
    print(f"[INFO] Add your images to: {sample_dir}/train/images/")
    print(f"[INFO] Add your labels to: {sample_dir}/train/labels/")
    return sample_dir


def convert_voc_to_yolo_seg(voc_annotation_path, img_width, img_height):
    """
    Convert VOC XML annotation to YOLO segmentation format.
    
    Since RDD2022 only has bounding boxes, we convert each bbox to 
    a polygon (4-point rectangle). For real segmentation, proper polygon
    annotations should be used.
    
    YOLO-seg format per line:
      class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0-1)
    """
    tree = ET.parse(voc_annotation_path)
    root = tree.getroot()
    
    labels = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text.strip()
        
        if class_name not in CLASS_MAP:
            continue
        
        class_id = CLASS_MAP[class_name]
        bbox = obj.find('bndbox')
        
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Clamp to image bounds
        xmin = max(0, min(xmin, img_width))
        ymin = max(0, min(ymin, img_height))
        xmax = max(0, min(xmax, img_width))
        ymax = max(0, min(ymax, img_height))
        
        if xmax <= xmin or ymax <= ymin:
            continue
        
        # Normalize coordinates to [0, 1]
        x1_n = xmin / img_width
        y1_n = ymin / img_height
        x2_n = xmax / img_width
        y2_n = ymax / img_height
        
        # Create polygon from bbox (4 corners for segmentation format)
        # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        polygon = f"{class_id} {x1_n:.6f} {y1_n:.6f} {x2_n:.6f} {y1_n:.6f} {x2_n:.6f} {y2_n:.6f} {x1_n:.6f} {y2_n:.6f}"
        labels.append(polygon)
    
    return labels


def convert_yolo_bbox_to_seg(bbox_label):
    """
    Convert YOLO bbox line (class x y w h) to YOLO seg line (class x1 y1 x2 y2 x3 y3 x4 y4)
    """
    parts = bbox_label.strip().split()
    if len(parts) != 5:
        return None
    
    class_id = parts[0]
    x_c, y_c, w, h = map(float, parts[1:])
    
    # Calculate corners (normalized)
    x1 = max(0.0, x_c - w/2)
    y1 = max(0.0, y_c - h/2)
    x2 = min(1.0, x_c + w/2)
    y2 = y1
    x3 = x2
    y3 = min(1.0, y_c + h/2)
    x4 = x1
    y4 = y3
    
    return f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"


def prepare_dataset(raw_dir, output_dir, val_split=0.2):
    """
    Prepare the dataset in YOLO segmentation format.
    
    Handles both:
    1. RDD2022 VOC format (XML annotations)
    2. Kaggle RDD2022 YOLO bbox format (txt labels)
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    
    # Check if we have the Kaggle structure (RDD_SPLIT)
    kaggle_split = raw_dir / 'RDD_SPLIT'
    
    if kaggle_split.exists():
        print(f"[INFO] Detected Kaggle RDD_SPLIT structure")
        for split in ['train', 'val']:
            src_img_dir = kaggle_split / split / 'images'
            src_lbl_dir = kaggle_split / split / 'labels'
            
            dst_img_dir = output_dir / split / 'images'
            dst_lbl_dir = output_dir / split / 'labels'
            
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            images = list(src_img_dir.glob('*.jpg')) + list(src_img_dir.glob('*.png'))
            print(f"[INFO] Processing {len(images)} images for {split}...")
            
            for img_path in images:
                # Copy image
                dst_img = dst_img_dir / img_path.name
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
                
                # Convert label
                lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
                dst_lbl = dst_lbl_dir / f"{img_path.stem}.txt"
                
                if lbl_path.exists():
                    with open(lbl_path, 'r') as f:
                        lines = f.readlines()
                    
                    seg_lines = []
                    for line in lines:
                        seg_line = convert_yolo_bbox_to_seg(line)
                        if seg_line:
                            seg_lines.append(seg_line)
                    
                    if seg_lines:
                        with open(dst_lbl, 'w') as f:
                            f.write('\n'.join(seg_lines))
        
        print(f"[INFO] Dataset prepared at {output_dir}")
        create_dataset_yaml(output_dir)
        return
        
    # Original logic for VOC/flat structure
    # Create output structure
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Find image and annotation directories
    img_dirs = list(raw_dir.rglob('images'))
    ann_dirs = list(raw_dir.rglob('annotations'))
    
    if not img_dirs:
        # Try flat structure
        img_dirs = [raw_dir]
    
    # Collect all images
    all_images = []
    for img_dir in img_dirs:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            all_images.extend(img_dir.glob(ext))
    
    if not all_images:
        print(f"[WARNING] No images found in {raw_dir}")
        print(f"[INFO] Expected structure: {raw_dir}/train/images/*.jpg")
        return
    
    print(f"[INFO] Found {len(all_images)} images")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - val_split))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"[INFO] Train: {len(train_images)}, Val: {len(val_images)}")
    
    for split, images in [('train', train_images), ('val', val_images)]:
        for img_path in images:
            # Copy image
            dst_img = output_dir / split / 'images' / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            
            # Look for annotation
            stem = img_path.stem
            
            # Try XML annotation (VOC format)
            xml_candidates = list(raw_dir.rglob(f'{stem}.xml'))
            txt_candidates = list(raw_dir.rglob(f'{stem}.txt'))
            
            dst_label = output_dir / split / 'labels' / f'{stem}.txt'
            
            if xml_candidates:
                # Convert VOC XML to YOLO seg format
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    w, h = img.size
                except:
                    w, h = 640, 640  # default
                
                labels = convert_voc_to_yolo_seg(xml_candidates[0], w, h)
                if labels:
                    with open(dst_label, 'w') as f:
                        f.write('\n'.join(labels))
                        
            elif txt_candidates:
                # Already in YOLO format — copy as-is
                shutil.copy2(txt_candidates[0], dst_label)
    
    print(f"[INFO] Dataset prepared at {output_dir}")
    
    # Create dataset YAML
    create_dataset_yaml(output_dir)


def create_dataset_yaml(data_dir):
    """Create the dataset.yaml file required by Ultralytics."""
    data_dir = Path(data_dir).resolve()
    
    yaml_content = f"""# RDD2022 Dataset Configuration for YOLOv8-PD-Seg
# ================================================
# Road Damage Detection 2022 — converted to segmentation format

path: {data_dir}
train: train/images
val: val/images

# Classes (4 road damage types)
nc: 4
names:
  0: D00_Longitudinal_Crack
  1: D10_Transverse_Crack
  2: D20_Alligator_Crack
  3: D40_Pothole
"""
    
    yaml_path = data_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"[INFO] Dataset YAML created: {yaml_path}")
    return yaml_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare RDD2022 dataset for YOLOv8-PD-Seg')
    parser.add_argument('--raw_dir', type=str, default=None, help='Path to raw dataset (contains train/val splits)')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to save processed dataset')
    args = parser.parse_args()

    data_root = Path(__file__).parent / 'datasets'
    
    print("=" * 60)
    print("RDD2022 Dataset Preparation (Kaggle Version)")
    print("=" * 60)
    
    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        # Point directly to the extracted Kaggle dataset
        raw_dir = data_root / 'raw' / 'RDD2022'
    
    if not raw_dir.exists():
        print(f"[ERROR] Extracted dataset not found at {raw_dir}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_root / 'rdd2022_seg'

    # Step 2: Convert to YOLO seg format
    prepare_dataset(raw_dir, output_dir, val_split=0.2)
    
    print("\n" + "=" * 60)
    print("Done! Dataset is ready for training.")
    print(f"Dataset path: {output_dir}")
    print(f"Config file:  {output_dir / 'dataset.yaml'}")
    print("=" * 60)
