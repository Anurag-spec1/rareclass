import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import config

def get_model(num_classes=config.NUM_CLASSES, freeze_backbone=False):
    """
    Initializes a DeepLabV3 model with a ResNet50 backbone.
    
    Args:
        num_classes (int): Number of output classes.
        freeze_backbone (bool): If True, freezes the parameters of the ResNet50 backbone 
                                to only train the classifier head.
                                
    Returns:
        torch.nn.Module: The configured DeepLabV3 model.
    """
    # Load a pre-trained DeepLabV3 model
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Replace the classifier head for the target number of classes
    # DeepLabV3 ResNet50 has an 'out_channels' of 2048 from the backbone
    model.classifier = DeepLabHead(2048, num_classes)
    
    # Optional: Also replace the aux classifier if it exists
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        from torchvision.models.segmentation.fcn import FCNHead
        # ResNet50 aux out_channels is 1024
        model.aux_classifier = FCNHead(1024, num_classes)

    return model

if __name__ == "__main__":
    # Quick verification of the model structure
    import logging
    logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info("Initializing model...")
    
    net = get_model(num_classes=config.NUM_CLASSES)
    dummy_input = torch.randn(1, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    
    logger.info("Running dummy forward pass...")
    output = net(dummy_input)
    
    # output is an OrderedDict with 'out' and optionally 'aux'
    logger.info(f"Output shape: {output['out'].shape}") 
    # Expected: [1, NUM_CLASSES, HEIGHT, WIDTH]
