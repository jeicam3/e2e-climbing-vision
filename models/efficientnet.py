import torch.nn as nn
from torchvision import models

def get_climbing_model(model_name="b0", num_classes=4, dropout_rate=0.6, freeze_until_block=8):
    """
    Uniwersalna funkcja tworząca modele z rodziny EfficientNet (B0, B1, B2).
    Zwraca krotkę: (model, native_resolution)
    
    Args:
        model_name (str): 'b0', 'b1' lub 'b2'
        num_classes (int): Liczba klas wyjściowych (domyślnie 4)
        dropout_rate (float): Współczynnik dropoutu w klasyfikatorze
        freeze_until_block (int): Indeks bloku, do którego włącznie zamrażamy wagi (0-8)
    """
    model_name = model_name.lower()
    
    config = {
        "b0": {
            "creator": models.efficientnet_b0,
            "weights": models.EfficientNet_B0_Weights.DEFAULT,
            "resolution": 224
        },
        "b1": {
            "creator": models.efficientnet_b1,
            "weights": models.EfficientNet_B1_Weights.DEFAULT,
            "resolution": 240
        },
        "b2": {
            "creator": models.efficientnet_b2,
            "weights": models.EfficientNet_B2_Weights.DEFAULT,
            "resolution": 260
        }
    }
    
    if model_name not in config:
        raise ValueError(f"Nieobsługiwany model: {model_name}. Wybierz spośród: {list(config.keys())}")
        
    cfg = config[model_name]
    
    model = cfg["creator"](weights=cfg["weights"])
    resolution = cfg["resolution"]
    
    print(f"Załadowano EfficientNet-{model_name.upper()} (Natywna rozdzielczość: {resolution}x{resolution})")
    
    #struktura bloków 0-8 jest identyczna w B0, B1, B2
    for i, block in enumerate(model.features):
        if i < freeze_until_block:
            for param in block.parameters():
                param.requires_grad = False
        else:
            for param in block.parameters():
                param.requires_grad = True
                
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout_rate), 
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid()
    )
    
    return model, resolution