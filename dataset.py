import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging

import config

logger = logging.getLogger(__name__)

class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for Semantic Segmentation.
    Includes logic for rare class oversampling by scanning masks during initialization
    and duplicating image/mask paths to balance the dataset.
    """
    def __init__(self, images_dir, masks_dir, transform=None, is_train=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.is_train = is_train

        # Safely load valid images (handling corrupted files)
        self.image_paths = []
        self.mask_paths = []
        
        if self.images_dir.exists() and self.masks_dir.exists():
            valid_extensions = ('.png', '.jpg', '.jpeg')
            for img_path in sorted(self.images_dir.iterdir()):
                if img_path.suffix.lower() in valid_extensions:
                    mask_path = self.masks_dir / f"{img_path.stem}.png"
                    if mask_path.exists():
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                    else:
                        logger.warning(f"Mask not found for image: {img_path.name}")
        else:
            logger.warning(f"Dataset directories do not exist: {images_dir} or {masks_dir}")

        # Rare Class Oversampling Logic (Only apply during training)
        if self.is_train and len(self.image_paths) > 0:
            self._apply_oversampling()

    def _apply_oversampling(self):
        """
        Scans all masks in the dataset to identify those containing rare classes.
        Duplicates the paths to artificially balance the dataset according to config.
        """
        logger.info("Scanning masks for rare class oversampling... This may take a moment.")
        
        rare_class_counts = {k: 0 for k in config.RARE_CLASS_TARGETS.keys()}
        rare_class_paths = {k: [] for k in config.RARE_CLASS_TARGETS.keys()}

        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            try:
                # Load mask in grayscale to check class IDs
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                
                unique_classes = np.unique(mask)
                for class_id in config.RARE_CLASS_TARGETS.keys():
                    if class_id in unique_classes:
                        rare_class_counts[class_id] += 1
                        rare_class_paths[class_id].append((img_path, mask_path))
            except Exception as e:
                logger.error(f"Error reading mask {mask_path} during oversampling scan: {e}")
        
        # Duplicate images to meet target counts
        for class_id, target_info in config.RARE_CLASS_TARGETS.items():
            target_count = target_info["target_count"]
            current_count = rare_class_counts[class_id]
            class_name = target_info["name"]
            
            if current_count > 0 and current_count < target_count:
                deficit = target_count - current_count
                logger.info(f"Oversampling class '{class_name}' (ID: {class_id}): "
                            f"Found {current_count}, Target {target_count}. Adding {deficit} duplicates.")
                
                # Randomly sample from available rare class images to meet the deficit
                indices = np.random.choice(len(rare_class_paths[class_id]), deficit, replace=True)
                for idx in indices:
                    img_path, mask_path = rare_class_paths[class_id][idx]
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
            else:
                logger.info(f"Class '{class_name}' (ID: {class_id}): Found {current_count}, Target {target_count}. No oversampling needed.")
        
        logger.info(f"Oversampling complete. Total training samples: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        mask_path = str(self.mask_paths[idx])

        # Load image and mask
        image = cv2.imread(img_path)
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            # Fallback to zero tensor to prevent crash
            image = np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error(f"Failed to load mask: {mask_path}")
            mask = np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH), dtype=np.uint8)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Basic fallback transformations if none provided
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

def get_transforms(is_train=True):
    """
    Returns albumentations transforms for training or validation.
    """
    if is_train:
        return A.Compose([
            A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def create_dataloaders():
    """
    Creates and returns train and validation dataloaders.
    """
    train_dataset = SegmentationDataset(
        images_dir=config.TRAIN_IMAGES_DIR,
        masks_dir=config.TRAIN_MASKS_DIR,
        transform=get_transforms(is_train=True),
        is_train=True
    )
    
    val_dataset = SegmentationDataset(
        images_dir=config.VAL_IMAGES_DIR,
        masks_dir=config.VAL_MASKS_DIR,
        transform=get_transforms(is_train=False),
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True if len(train_dataset) > config.BATCH_SIZE else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Setup basic logging for testing the dataset module directly
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Dataset module...")
    train_loader, val_loader = create_dataloaders()
    logger.info("Dataloaders initialized successfully.")
