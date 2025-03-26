# Real-Time Object Detection with Dynamic Resource Allocation for Advanced Driver Assistance Systems (ADAS)

This project enhances Advanced Driver Assistance Systems by implementing a dynamic object detection framework that intelligently switches between YOLOv8 model variants based on real-time system conditions.

## 🎯 Key Problem Addressed

Traditional object detection systems struggle with varying computational resources and dynamic driving conditions. DynamicDet solves this by:

- Dynamically selecting optimal YOLO models
- Adapting to real-time system constraints
- Ensuring consistent performance across edge devices

## 🚀 Features

- **Adaptive Model Selection:** Switches between YOLOv8-Small and YOLOv8-Nano models
- **Resource-Aware Detection:** Optimizes performance based on:
  - Computational resources
  - Vehicle speed
  - Battery level
- **Event-Driven Pipeline:** Supports real-time triggers for:
  - Collision warnings
  - Pedestrian alerts
  - Lane assistance
- **Edge Device Compatibility:** Optimized for Jetson Nano, Raspberry Pi, and similar platforms

## 📦 Installation

### Setup

1. Clone the repository:
```bash
git clone https://github.com/VDIGPKU/DynamicDet.git
cd DynamicDet
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

This project uses the BDD100K dataset for realistic ADAS scenarios.

### Dataset Structure
```
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
```

### Configure Dataset
Edit `data/dataset.yaml`:
```yaml
train: datasets/images/train
val: datasets/images/val
nc: 3
names: ['car', 'person', 'traffic_sign']
```

## 🏋️ Training

Train the dynamic object detection model:
```bash
python train.py \
    --data data/dataset.yaml \
    --cfg configs/yolov8_dynamic.yaml \
    --weights yolov8n.pt \
    --epochs 100 \
    --batch-size 16
```

## 🔍 Inference

Run real-time object detection:
```bash
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source 0  # 0 for webcam, or provide video/image path
```

## 🧠 Dynamic Model Selection

Example of intelligent model switching:
```python
from ultralytics import YOLO

def dynamic_model_selection(speed):
    if speed > 80:
        return 'yolov8n.pt'  # Lightweight for high speed
    elif speed > 40:
        return 'yolov8s.pt'  # Balanced model
    else:
        return 'yolov8m.pt'  # Accurate for low speed

model_path = dynamic_model_selection(vehicle_speed)
model = YOLO(model_path)
results = model(source=0, show=True)
```

## 🚧 Neural Architecture Search (NAS)

Automatically optimize model architecture:
```bash
python nas.py \
    --data data/dataset.yaml \
    --cfg configs/yolov8_dynamic.yaml
```

## 📌 Future Development

- Expand hardware compatibility
- Improve dynamic model selection algorithms
- Integrate with more ADAS platforms
- Enhance hardware resource monitoring
- Develop advanced neural architecture search techniques


