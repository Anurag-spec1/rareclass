# Rare Class Oversampling - Semantic Segmentation

This project implements a PyTorch-based semantic segmentation pipeline using a DeepLabV3 architecture with a ResNet50 backbone. The core objective is to address class imbalance through **custom dataloader logic** that artificially duplicates images containing rare classes (specifically "Flowers" and "Lush Bushes").

## Table of Contents
1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training the Model](#training-the-model)
5. [Inference and Evaluation](#inference-and-evaluation)

## Project Structure
```text
.
├── config.py       # Centralized hyperparameters, paths, and class configurations.
├── dataset.py      # Custom PyTorch Dataset with rare class oversampling logic.
├── model.py        # DeepLabV3 model architecture definition.
├── train.py        # Training loop with AMP, logging, and checkpointing.
├── test.py         # Inference script for evaluation (mIoU, latency, confusion matrix).
└── README.md       # Project documentation.
```

## Environment Setup
It is highly recommended to use a virtual environment (e.g., `conda` or `venv`).

1. Create and activate a new environment:
   ```bash
   conda create -n seg_env python=3.9 -y
   conda activate seg_env
   ```
2. Install the required dependencies:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install numpy opencv-python albumentations scikit-learn tqdm
   ```

## Dataset Preparation
1. Ensure your dataset is structured according to the paths defined in `config.py`:
   ```text
   data/
   ├── train/
   │   ├── images/
   │   └── masks/
   ├── val/
   │   ├── images/
   │   └── masks/
   └── test/
       ├── images/
       └── masks/
   ```
2. Masks should be saved as grayscale PNG images where the pixel value corresponds to the class ID (0-9 for 10 classes). Background is typically 0, Flowers 8, and Lush Bushes 9.

## Training the Model
To start training the model with the oversampling logic enabled:

```bash
python train.py
```

- **Features Used:** 
  - PyTorch Automatic Mixed Precision (AMP) is enabled for faster training on supported GPUs.
  - The `dataset.py` automatically scans the training set on initialization and duplicates image/mask pairs that contain rare classes until they reach the counts specified in `config.py` (`RARE_CLASS_TARGETS`).
- Checkpoints are saved under `outputs/checkpoints/best_model.pth`.
- Training logs are saved to `outputs/logs/train.log`.

To optionally freeze the ResNet50 backbone and only train the DeepLabV3 classification head:
```bash
python train.py --freeze_backbone
```

## Inference and Evaluation
To evaluate the trained model on the test dataset:

```bash
python test.py
```

This script will:
1. Load `best_model.pth`.
2. Perform inference image-by-image.
3. Use `time.perf_counter()` to strictly measure **execution time per image** to verify latency constraints.
4. Output a detailed **Confusion Matrix** to ensure that rare classes (Flowers, Lush Bushes) are successfully identified without heavily biasing towards background classes.
5. Log final per-class IoU and overall mean IoU (mIoU).

> **Optimization Note:** If inference time strictly exceeds a <50ms constraint on your target hardware, consider modifying `model.py` to use a lighter backbone (e.g., `MobileNetV3` instead of `ResNet50`), or export the PyTorch model to `TensorRT`/`ONNX` for hardware-accelerated execution.
