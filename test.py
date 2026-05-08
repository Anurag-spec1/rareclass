import argparse
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from model import get_model
from dataset import SegmentationDataset

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_DIR / "test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# UTILITIES
# ==============================================================================
def get_test_transforms():
    return A.Compose([
        A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def print_confusion_matrix(cm, class_names):
    """Prints a nicely formatted confusion matrix to the logger."""
    logger.info("=== Confusion Matrix ===")
    
    # Calculate per-class metrics
    row_sums = cm.sum(axis=1)
    
    # Header
    header = f"{'Class':<15} | " + " | ".join([f"{name[:3]:>4}" for name in class_names])
    logger.info(header)
    logger.info("-" * len(header))
    
    for i, name in enumerate(class_names):
        row_str = f"{name[:15]:<15} | "
        for j in range(len(class_names)):
            val = cm[i, j]
            row_str += f"{val:4d} | "
        
        # Calculate recall (accuracy per class)
        recall = cm[i, i] / row_sums[i] if row_sums[i] > 0 else 0
        row_str += f" (Recall: {recall:.2f})"
        logger.info(row_str)
        
def calculate_iou_per_class(cm):
    """Calculates IoU for each class from the confusion matrix."""
    ious = []
    for i in range(cm.shape[0]):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

# ==============================================================================
# INFERENCE PIPELINE
# ==============================================================================
def evaluate(args):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Model
    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    logger.info("Loading model...")
    model = get_model(num_classes=config.NUM_CLASSES)
    
    # Load checkpoint safely
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 2. Setup Test Dataset
    test_dataset = SegmentationDataset(
        images_dir=config.TEST_IMAGES_DIR,
        masks_dir=config.TEST_MASKS_DIR,
        transform=get_test_transforms(),
        is_train=False
    )
    
    # Note: We process images 1 by 1 (batch_size=1) to accurately measure per-image inference time
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=1
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")

    # 3. Inference Loop
    all_preds = []
    all_labels = []
    total_time = 0.0
    
    # Warmup for accurate timing on GPU
    logger.info("Warming up model...")
    if len(test_dataset) > 0:
        with torch.no_grad():
            dummy_input, _ = test_dataset[0]
            dummy_input = dummy_input.unsqueeze(0).to(device)
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    logger.info("Starting evaluation...")
    with torch.no_grad():
        for idx, (image, mask) in enumerate(test_loader):
            image = image.to(device)
            mask_np = mask.squeeze().numpy()
            
            # Start timer
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Forward pass
            output = model(image)
            logits = output['out']
            preds = torch.argmax(logits, dim=1).squeeze(0)
            
            # End timer
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            inference_time = end_time - start_time
            total_time += inference_time
            
            # Save predictions and labels for metrics
            pred_np = preds.cpu().numpy()
            
            # Flatten to 1D for confusion matrix
            valid_idx = mask_np != 255 # Ignore index if any
            all_preds.append(pred_np[valid_idx])
            all_labels.append(mask_np[valid_idx])
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(test_loader)} images...")

    # 4. Calculate Metrics
    avg_time_ms = (total_time / len(test_loader)) * 1000
    logger.info(f"[*] Average Inference Time: {avg_time_ms:.2f} ms per image")
    
    # Explicit documentation regarding the latency constraint:
    # If the constraint is strictly < 50ms per image and this ResNet50 DeepLabV3 model 
    # exceeds it on the target hardware, consider using a lighter backbone (e.g., MobileNetV3) 
    # or utilizing optimization libraries like TensorRT or ONNX Runtime.

    logger.info("Calculating metrics...")
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    cm = confusion_matrix(all_labels, all_preds, labels=range(config.NUM_CLASSES))
    print_confusion_matrix(cm, config.CLASS_NAMES)
    
    ious = calculate_iou_per_class(cm)
    logger.info("=== Per-Class IoU ===")
    for name, iou in zip(config.CLASS_NAMES, ious):
        logger.info(f"{name:<15}: {iou:.4f}")
        
    mIoU = np.nanmean(ious)
    logger.info(f"[*] Mean IoU (mIoU): {mIoU:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Semantic Segmentation Model")
    args = parser.parse_args()
    
    evaluate(args)
