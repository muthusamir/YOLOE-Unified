# YOLOE-Unified: Real-Time Promptable Detection and Segmentation on Edge Devices

**YOLOE-Unified** is a novel framework that integrates YOLOE with distilled CLIP, runtime SAM refinement, and TensorRT optimization for efficient open-vocabulary object detection and instance segmentation on edge devices (Jetson Orin, etc.).

### Highlights
- State-of-the-art zero-shot performance: **48.6% mAP** (detection), **42.8% mask AP** on LVIS
- Real-time on edge: **142 FPS (FP16)**, **118 FPS (INT8)** on Jetson Orin NX
- Low power: ~14W average consumption
- Supports text, visual, and prompt-free modes

### Installation
```bash
git clone https://github.com/yourusername/YOLOE-Unified.git
cd YOLOE-Unified
pip install -r requirements.txt
pip install -e .
