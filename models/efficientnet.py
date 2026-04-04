import torch.nn as nn
from torchvision import models

def get_climbing_model(num_classes=4, dropout_rate=0.6):
    #bazowy model - EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Wyciągnięcie liczby cech wejściowych do klasyfikatora
    num_ftrs = model.classifier[1].in_features
    
    #zamiana klasyfikatora na specyfinczy dla wykrywania chwytów
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout_rate), 
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid()
    )
    
    return model