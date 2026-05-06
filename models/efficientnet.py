import torch.nn as nn
from torchvision import models

def get_climbing_model(num_classes=4, dropout_rate=0.6, mode='fine_tune_top'):
    """
        mode (str): 
            'freeze_all' -> Trenujemy tylko klasyfikator
            'fine_tune_top' -> Klasyfikator + ostatni blok features[8]
            'full_train' -> Wszystkie warstwy odblokowane
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    #domyślnie wszystko zamrożone
    for param in model.parameters():
        param.requires_grad = False
    
    if mode == 'full_train':
        for param in model.parameters():
            param.requires_grad = True
        
    elif mode == 'fine_tune_top':
        # odmrożony ostatni blok konwolucyjny
        for param in model.features[8].parameters():
            param.requires_grad = True

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout_rate), 
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid()
    )
    
    return model