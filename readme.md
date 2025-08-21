# exp-pytorch-circle-or-not

Tiny PyTorch project that lets you **draw a shape** and tells you **YES (circle)** or **NO (not circle)** with a confidence score.  
Training data is synthetic (wobbly circles, mild ovals, open arcs, squares, triangles). Drawings are **crop → pad → resize (32×32)** so position/size don’t matter.

## Files
- `generate_data.py` – builds `data/train` & `data/val` (circle / not_circle).
- `model_cnn.py` – small CNN (2 conv blocks → 2 linear layers).
- `train.py` – trains and saves `models/circle_vs_not.pth`.
- `classify_image.py` – **draw with mouse; release to classify** (auto-clear after a few seconds).
