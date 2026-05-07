import torch.nn as nn
from torchvision import models

def get_climbing_model(num_classes=4, dropout_rate=0.6, freeze_until_block=8):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # features to bloki od 0 do 8
    # 8 to ostatni blok przed klasyfikatorem
    for i, block in enumerate(model.features):
        if i < freeze_until_block:
            for param in block.parameters():
                param.requires_grad = False
        else:
            for param in block.parameters():
                param.requires_grad = True
                
    # klasyfikator zawsze odblokowany
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout_rate), 
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid()
    )
    return model