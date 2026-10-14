"""
Verify Custom Architecture
==========================
This script verifies that the custom YOLOv8-PD-Seg architecture
builds correctly with our custom modules (C2fGhost, BOTModule, LSKAModule).

It counts the parameters and runs a dummy forward pass to ensure
the graph connects properly.
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def verify_architecture():
    # 1. Register modules
    print("=" * 60)
    print("1. Registering Custom Modules...")
    from register_modules import register_custom_modules
    register_custom_modules()

    # 2. Build model from our custom YAML
    print("\n" + "=" * 60)
    print("2. Building YOLOv8-PD-Seg from configs/yolov8-pd-seg.yaml...")
    from ultralytics import YOLO
    
    try:
        model = YOLO('configs/yolov8-pd-seg.yaml')
        print("[OK] Model built successfully.")
    except Exception as e:
        print(f"[ERROR] Model build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Print Summary
    print("\n" + "=" * 60)
    print("3. Architecture Summary")
    
    # We can inspect the model.model layers
    layers = list(model.model.model)
    print(f"Total layers: {len(layers)}")
    
    modules_found = {'C2fGhost': 0, 'BOTModule': 0, 'LSKAModule': 0}
    
    for i, m in enumerate(layers):
        m_type = type(m).__name__
        if m_type in modules_found:
            modules_found[m_type] += 1
            print(f"  Layer {i:2d}: {m_type:15s} (Params: {sum(p.numel() for p in m.parameters()):,})")
            
    print("\nCustom Module Counts:")
    for k, v in modules_found.items():
        print(f"  - {k}: {v} (Expected: C2fGhost=7, BOT=1, LSKA=3)")
        
    # Get total parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"(For reference: Vanilla YOLOv8n-seg has ~3.2M params)")
    
    # 4. Dummy forward pass
    print("\n" + "=" * 60)
    print("4. Testing Forward Pass...")
    model.model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    try:
        with torch.no_grad():
            output = model.model(dummy_input)
        print(f"[OK] Forward pass successful!")
        
        # Output of SegmentationModel is typically (preds, features) or similar
        # For evaluation it returns [boxes, masks]
        if isinstance(output, tuple) or isinstance(output, list):
            print(f"Output shapes:")
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"  - Tensor {i}: {o.shape}")
                elif isinstance(o, tuple) or isinstance(o, list):
                    print(f"  - Tuple/List {i} with {len(o)} elements")
                    for j, item in enumerate(o):
                        if isinstance(item, torch.Tensor):
                            print(f"      - Item {j}: {item.shape}")
        
    except Exception as e:
        print(f"[ERROR] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\n[SUCCESS] Custom architecture is fully functional!")

if __name__ == '__main__':
    verify_architecture()
