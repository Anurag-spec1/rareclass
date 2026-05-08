import os
import cv2
import numpy as np
from pathlib import Path

def create_dummy_dataset(base_dir="data", num_train=10, num_val=2, num_test=2):
    base_dir = Path(base_dir)
    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test
    }
    
    height, width = 512, 512
    num_classes = 10
    
    for split, num_images in splits.items():
        img_dir = base_dir / split / "images"
        mask_dir = base_dir / split / "masks"
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        for i in range(num_images):
            # Create a dummy image (random noise)
            img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
            # Create a dummy mask (random classes 0 to 9)
            # Make sure rare classes (8 and 9) occasionally appear
            mask = np.random.randint(0, num_classes, (height, width), dtype=np.uint8)
            
            # Save
            cv2.imwrite(str(img_dir / f"dummy_{i:03d}.png"), img)
            cv2.imwrite(str(mask_dir / f"dummy_{i:03d}.png"), mask)
            
    print("Dummy dataset created successfully!")

if __name__ == "__main__":
    create_dummy_dataset()
