import os
from pathlib import Path

# ==============================================================================
# PROJECT CONFIGURATION
# ==============================================================================

# Project details
PROJECT_NAME = "Rare Class Oversampling - Semantic Segmentation"
CORE_ARCHITECTURE = "DeepLabV3 with ResNet50 Backbone"
DATASET_STRATEGY = "Full 10 classes with Rare Class Oversampling (Flowers, Lush Bushes)"

# ==============================================================================
# DATASET PARAMETERS
# ==============================================================================
# In a real scenario, base paths should be adjusted or passed via CLI
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_IMAGES_DIR = DATA_DIR / "train" / "images"
TRAIN_MASKS_DIR = DATA_DIR / "train" / "masks"
VAL_IMAGES_DIR = DATA_DIR / "val" / "images"
VAL_MASKS_DIR = DATA_DIR / "val" / "masks"
TEST_IMAGES_DIR = DATA_DIR / "test" / "images"
TEST_MASKS_DIR = DATA_DIR / "test" / "masks"

# Class Mappings (10 classes total)
NUM_CLASSES = 10
CLASS_NAMES = [
    "Background", 
    "Sky", 
    "Landscape", 
    "Water", 
    "Buildings",
    "Roads", 
    "Vehicles", 
    "Pedestrians", 
    "Flowers",       # Rare Class 1
    "Lush Bushes"    # Rare Class 2
]

# Oversampling Config
RARE_CLASS_TARGETS = {
    8: {"name": "Flowers", "target_count": 600},       # Class ID 8
    9: {"name": "Lush Bushes", "target_count": 200}    # Class ID 9
}

# Image parameters
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# ==============================================================================
# HYPERPARAMETERS & HARDWARE TOGGLES
# ==============================================================================
# Training hyperparams
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Hardware toggles
USE_AMP = True             # Automatic Mixed Precision
NUM_WORKERS = 4
PIN_MEMORY = True
DEVICE = "cuda"            # "cuda" or "cpu"

# ==============================================================================
# OUTPUT AND LOGGING
# ==============================================================================
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# Ensure output directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
