"""
Register Custom Modules with Ultralytics
==========================================
This module injects our custom modules (C2fGhost, BOTModule, LSKAModule)
into Ultralytics' module resolver so that they can be referenced by name
in YAML config files.

Ultralytics resolves module names via globals() in ultralytics.nn.tasks.parse_model().
We inject our modules there AND patch the base_modules/repeat_modules frozensets
so that Ultralytics correctly handles channel routing and depth scaling.

MUST be called BEFORE YOLO(yaml_path).

Usage:
    from register_modules import register_custom_modules
    register_custom_modules()
    model = YOLO('configs/yolov8-pd-seg.yaml')
"""

import sys
import inspect
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def register_custom_modules():
    """
    Inject custom modules into Ultralytics' global namespace so that
    parse_model() can resolve them from YAML config files and handle
    their channel routing properly.
    """
    from models.modules import C2fGhost, BOTModule, LSKAModule

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as nn_modules

    # ------------------------------------------------------------------
    # Step 1: Inject into tasks.py globals (where parse_model looks)
    # ------------------------------------------------------------------
    tasks.C2fGhost = C2fGhost
    tasks.BOTModule = BOTModule
    tasks.LSKAModule = LSKAModule

    # Also inject into nn.modules namespace for completeness
    nn_modules.C2fGhost = C2fGhost
    nn_modules.BOTModule = BOTModule
    nn_modules.LSKAModule = LSKAModule

    # ------------------------------------------------------------------
    # Step 2: Patch parse_model using metaprogramming
    # ------------------------------------------------------------------
    # We read the source of parse_model, add our modules to the
    # base_modules and repeat_modules frozensets, and re-execute it.
    
    src = inspect.getsource(tasks.parse_model)
    
    # 2.1 Add to base_modules
    if 'C2fGhost' not in src:
        # Find a known module in base_modules and append ours
        src = src.replace('C2fPSA,', 'C2fPSA,\n            C2fGhost,\n            BOTModule,\n            LSKAModule,')


    
    # 2.2 Add to repeat_modules
    if 'repeat_modules =' in src:
        # C2fGhost and BOTModule take 'n' (repeats) argument, LSKA doesn't
        src = src.replace('C2fPSA,', 'C2fPSA,\n            C2fGhost,\n            BOTModule,')

    # 2.3 Re-compile and overwrite parse_model in the tasks module
    exec(src, vars(tasks))
    
    print("[register_modules] Custom modules injected successfully:")
    print(f"  - C2fGhost")
    print(f"  - BOTModule")
    print(f"  - LSKAModule")

    return True


if __name__ == "__main__":
    register_custom_modules()
    print("\n[OK] Run successful.")
