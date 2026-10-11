import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import sys

# Import custom modules so YOLOv8 can load our custom architecture
try:
    from register_modules import register_custom_modules
    from models.depth_estimator import DepthEstimator
except ImportError:
    print("Error: Ensure you are running this script from the CV_Project root directory.")
    sys.exit(1)

def characterize_image(image_path, model_path, output_path=None):
    """
    Combines YOLOv8-PD-Seg (Segmentation) with Depth Estimation
    to characterize potholes (Area, Depth, Volume).
    """
    print(f"\n🚀 Processing: {image_path}")
    
    # 1. Register our custom modules (BOT, LSKA, C2fGhost, etc.)
    register_custom_modules()
    
    # 2. Load the trained Segmentation Model
    print("Loading YOLOv8-PD-Seg model...")
    seg_model = YOLO(model_path)
    
    # 3. Load the Depth Estimator (Paper 2 Approach)
    print("Loading Depth Estimator...")
    depth_model = DepthEstimator(model_type="MiDaS_small")
    depth_model.load_model()
    
    # 4. Read Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    original_img = img.copy()
    h, w = img.shape[:2]
    
    # 5. Run Segmentation Inference
    print("Running Segmentation...")
    results = seg_model(img, conf=0.25)
    result = results[0]
    
    if result.masks is None:
        print("No potholes or road damage detected in the image.")
        cv2.imwrite(output_path or "output_no_damage.jpg", original_img)
        return
        
    # 6. Run Depth Estimation
    print("Running Depth Estimation...")
    depth_map = depth_model.estimate_depth(original_img)
    
    # Create an overlay image
    overlay = original_img.copy()
    
    # Colors for different classes (BGR)
    colors = {
        0: (255, 150, 0),    # D00 Longitudinal
        1: (0, 255, 255),    # D10 Transverse
        2: (255, 0, 255),    # D20 Alligator
        3: (0, 0, 255)       # D40 Pothole (Red)
    }
    
    print(f"\n📊 Characterization Results:")
    print("-" * 50)
    
    # 7. Process each detected instance
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.data.cpu().numpy()
    
    # Resize masks to original image size
    masks = np.array([cv2.resize(m, (w, h)) for m in masks])
    
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        class_id = int(box[5])
        conf = float(box[4])
        class_name = result.names[class_id]
        
        # Color for this class
        color = colors.get(class_id, (0, 255, 0))
        
        # Draw Mask Overlay
        binary_mask = (mask > 0.5).astype(np.uint8)
        colored_mask = np.zeros_like(original_img)
        colored_mask[binary_mask == 1] = color
        
        # Blend mask with image
        cv2.addWeighted(colored_mask, 0.5, overlay, 1.0, 0, overlay)
        
        # Draw Bounding Box
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # ONLY characterize depth/volume if it's a Pothole (class 3)
        if class_id == 3: # D40_Pothole
            # Extract depth stats for this pothole
            depth_stats = depth_model.extract_pothole_depth(depth_map, binary_mask)
            
            # Simple volume estimation (pixels * relative depth)
            # In a real scenario with calibrated cameras, this would be in cubic meters
            relative_vol = depth_stats['area_pixels'] * depth_stats['relative_depth']
            
            print(f"Pothole {i+1} [Conf: {conf:.2f}]:")
            print(f"  - Area (Pixels): {depth_stats['area_pixels']}")
            print(f"  - Avg Depth: {depth_stats['mean_depth']:.4f}")
            print(f"  - Max Depth: {depth_stats['max_depth']:.4f}")
            print(f"  - Volume Est: {relative_vol:.2f} units")
            
            # Add text to image
            label = f"Pothole Vol: {relative_vol:.0f}"
            cv2.putText(overlay, label, (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            print(f"Damage {i+1} ({class_name}) [Conf: {conf:.2f}]")
            label = f"{class_name} {conf:.2f}"
            cv2.putText(overlay, label, (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 8. Create a side-by-side visualization (Original + Depth + Segmented)
    print("\nSaving visualization...")
    
    # Convert depth map to colormap for visualization
    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
    # Stack images horizontally
    # Resize all to same height if necessary, but here they are same size
    combined = np.hstack((original_img, overlay, depth_color))
    
    # Add titles
    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Seg + Charact", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Depth Map", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if output_path is None:
        output_path = f"characterization_output_{Path(image_path).name}"
        
    cv2.imwrite(output_path, combined)
    print(f"✅ Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pothole Characterization (Seg + Depth)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="runs/segment/improved_yolov8_pd_seg/weights/best.pt", help="Path to trained YOLOv8-PD-Seg model")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first or provide the correct path.")
        sys.exit(1)
        
    characterize_image(args.image, args.model, args.output)
