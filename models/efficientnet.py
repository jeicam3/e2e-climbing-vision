import torch.nn as nn
from torchvision import models

def get_climbing_model(num_classes=4, dropout_rate=0.6, freeze_backbone=False):
    """Tworzy model EfficientNet-B0 z opcją zamrożenia wag warstw bazowych."""
    # pobranie bazowego modelu z wagami ImageNet
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    #opcjonalne zamrożenie wag warstw bazowych - do E2E
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # wyciągnięcie liczby cech wejściowych
    num_ftrs = model.classifier[1].in_features
    
    # nadpisanie klasyfikatora
    # warstwy wewnątrz nn.Sequential domyślnie mają requires_grad=True
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout_rate), 
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid()
    )
    
    return model