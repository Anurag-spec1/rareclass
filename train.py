import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import config
from dataset import create_dataloaders
from model import get_model

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_DIR / "train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# UTILITIES
# ==============================================================================
def set_seed(seed=42):
    """Ensures reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")

def calculate_iou(preds, labels, num_classes):
    """
    Calculates intersection over union (IoU) for a batch.
    Args:
        preds: Tensor of shape (B, H, W) with predicted class IDs
        labels: Tensor of shape (B, H, W) with ground truth class IDs
    Returns:
        Mean IoU ignoring background or purely average.
    """
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    # Ignore index 255 if it exists in your dataset mapping (e.g. padding/ignore)
    valid_mask = (labels != 255)
    preds = preds[valid_mask]
    labels = labels[valid_mask]

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union > 0:
            ious.append(float(intersection) / float(max(union, 1)))
            
    return np.mean(ious) if ious else 0.0

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================
def train(args):
    set_seed(args.seed)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Initialize Dataloaders
    logger.info("Initializing dataloaders...")
    train_loader, val_loader = create_dataloaders()
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 2. Initialize Model
    logger.info("Initializing model...")
    model = get_model(num_classes=config.NUM_CLASSES, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # 3. Loss and Optimizer
    # CrossEntropyLoss automatically handles class probabilities. 
    # Can consider weights here if oversampling wasn't enough, but we rely on dataloader oversampling.
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.USE_AMP)

    best_val_iou = 0.0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # 4. Training Loop
    logger.info("Starting training loop...")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # tqdm for visual progress if running interactively
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]")
        
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, dtype=torch.long, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast(enabled=config.USE_AMP):
                outputs = model(images)
                # DeepLabV3 returns an OrderedDict with 'out'
                logits = outputs['out']
                loss = criterion(logits, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Val]")
            for images, masks in pbar_val:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, dtype=torch.long, non_blocking=True)
                
                with autocast(enabled=config.USE_AMP):
                    outputs = model(images)
                    logits = outputs['out']
                    loss = criterion(logits, masks)
                
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                iou = calculate_iou(preds, masks, config.NUM_CLASSES)
                val_iou += iou
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        logger.info(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_iou:.4f}")

        # 6. Checkpoint Saving
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
            }, checkpoint_path)
            logger.info(f"[*] New best model saved to {checkpoint_path} with mIoU: {best_val_iou:.4f}")

    logger.info("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Semantic Segmentation Model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze the ResNet backbone")
    args = parser.parse_args()
    
    train(args)
