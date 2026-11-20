# LaTeX Research Paper — YOLOv8-PD-Seg

## How to Use on Overleaf

### Quick Setup (Recommended)
1. Go to [Overleaf](https://www.overleaf.com)
2. Click **New Project → Upload Project**
3. Upload this entire `latex_paper/` folder as a `.zip`
4. Set the main document to `main.tex`
5. The IEEE `IEEEtran.cls` class is **already built into Overleaf** — no need to upload it separately
6. Click **Recompile**

### Adding Figures
To include the project's result images, copy them into the `figures/` subdirectory:
```
latex_paper/
├── main.tex          ← Main LaTeX source
├── README.md         ← This file
└── figures/          ← Place images here
    ├── benchmark_comparison.jpg
    ├── architecture_improvements.jpg
    ├── dual_pothole.jpg
    ├── car_wheel_pothole.jpg
    ├── water_pothole.jpg
    ├── road_crack.jpg
    ├── traffic_pothole.jpg
    ├── confusion_matrix.png
    ├── results.png
    └── ...
```

Then add `\includegraphics` commands in the LaTeX source where needed.

### Figures from your project (copy these in):
- `assets/comparisons/1_Benchmark_Accuracy_Comparison.jpg`
- `assets/comparisons/2_Architecture_Improvements_Card.jpg`
- `assets/characterization/*.jpg` (all 4 characterization panels)
- `runs/segment/runs/test_pipeline/confusion_matrix.png`
- `runs/segment/runs/test_pipeline/results.png`
- `runs/segment/runs/test_pipeline/BoxPR_curve.png`
- `runs/segment/runs/test_pipeline/MaskPR_curve.png`

### Template Info
- **Template:** IEEE Conference (IEEEtran class)
- **Compilation:** pdfLaTeX (default on Overleaf)
- **References:** Inline `thebibliography` (no external `.bib` file needed)

## Paper Contents
1. Abstract
2. Introduction (problem statement + our contributions)
3. Related Work (road damage detection, attention, segmentation, depth)
4. Methodology (architecture, backbone, seg head, depth, characterization)
5. Dataset (RDD2022 — 26,869 train, 5,758 val, 4 classes, 6 countries)
6. Experimental Setup (hyperparameters, augmentation, metrics)
7. Results and Discussion (benchmarks, characterization, ablation)
8. Conclusion and Future Work
9. References (19 citations)
