import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class ClimbingDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        labels = torch.tensor(self.annotations.iloc[index, 1:5].values.astype('float32'))
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

class ApplyTransform(Dataset):
    """Pomocniczy klasa do nakładania różnych transformacji"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)