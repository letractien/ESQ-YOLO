# ESQ-YOLO

## Project Overview

- `custom_ema_sqex.py`, `custom_ema.py`, `custom_sqex.py`: Experimental files with EMA, SqEx modules for YOLOv8.
- `ultralytics/` directory: Contains the main YOLOv8 source code and model configurations.
- `datasets/` directory: Contains sample data (e.g., BCCD).
- **SE (Squeeze-and-Excitation) and EMA (Efficient Multi-scale Attention) modules are pre-installed in the `ultralytics/nn/` directory.**

## System Requirements

- CPU: Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz
- RAM: 128GB+
- GPU: NVIDIA Quadro RTX 4000 (8GB VRAM)
- OS: Windows 10/11
- Python: 3.9.11
- CUDA: 12.4 (compatible with driver 551.61)

## 1. Environment Setup

1. **Install Python 3.9.11**  
   Download and install from [python.org](https://www.python.org/downloads/release/python-3911/).

2. **Install dependencies**  
   Open terminal/cmd in the project directory and run:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Data Preparation and Configuration

- **Download and extract the BCCD dataset:**
  - Visit: [https://www.kaggle.com/datasets/sefatenurkhandakar/blood-cell-detection-datatset](https://www.kaggle.com/datasets/sefatenurkhandakar/blood-cell-detection-datatset)
  - Log in and download the dataset zip file.
  - Extract all contents into the `datasets/bccd/` directory in the project (create the folder if it does not exist).

- **Copy the dataset config file:**
  - Copy `datasets/bccd/data.yaml` to the `ultralytics/yolo/data/datasets/` directory.
  - Make sure the paths in `ultralytics/yolo/data/datasets/data.yaml` point to the correct data locations (use absolute paths suitable for your machine). For example:
    ```yaml
    train: /absolute/path/to/your/project/datasets/bccd/train
    val: /absolute/path/to/your/project/datasets/bccd/valid
    test: /absolute/path/to/your/project/datasets/bccd/test
    ```
  - **Note:** The absolute path must be replaced with the correct data folder location on your machine to avoid errors during training.

## 3. Model Running Guide

**General procedure:**
1. Set up the environment (see section 1).
2. Prepare data and config files (see section 2).
3. Run the corresponding custom file for each model version.

**Note:**  
- Model config files are available in `ultralytics/models/v8/`.
- Sample data: `ultralytics/yolo/data/datasets/data.yaml` (or replace with your own yaml file).
- Change data paths if needed.

### Run custom files for each model version

#### 1. yolov8n + ema + sqex
Run:
```bash
python custom_ema_sqex.py
```

#### 2. yolov8n + sqex
Run:
```bash
python custom_sqex.py 
```

#### 3. yolov8n + ema
Run:
```bash
python custom_ema.py 
```

#### 4. yolov8m
Run:
```bash
python run_yolov8m.py
```

#### 5. yolov8s
Run:
```bash
python run_yolov8s.py
```

#### 6. yolov8n
Run:
```bash
python run_yolov8n.py
```

#### 7. yolov7-tiny
> **Note:** You need to install YOLOv7 separately, not included in this source.  
> Reference: [YOLOv7 GitHub](https://github.com/WongKinYiu/yolov7)

Run:
```bash
python train.py --workers 8 --device 0 --batch-size 16 --data ultralytics/yolo/data/datasets/data.yaml --img 640 --cfg cfg/training/yolov7-tiny.yaml --weights '' --name yolov7-tiny --cache --epochs 150
```

#### 8. yolov5n
> **Note:** You need to install YOLOv5 separately, not included in this source.  
> Reference: [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)

Run:
```bash
python train.py --workers 8 --device 0 --batch-size 16 --data ultralytics/yolo/data/datasets/data.yaml --img 640 --cfg yolov5n.yaml --weights '' --name yolov5n --cache --epochs 150
```

---

## Contact
Email: letractienw@gmail.com