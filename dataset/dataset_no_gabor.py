import os.path

import cv2
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize,Lambda
from torch.utils.data import DataLoader
from config import Config

class RescaleToUnitRange(torch.nn.Module):
    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min())
class GestureDataSetNoGabor(Dataset):
    def __init__(self, data_path: str, data: pd.DataFrame):
        super(GestureDataSetNoGabor).__init__()
        self.config = Config()
        self.data_path = data_path
        self.data = data['num']
        self.labels = data['label']
        self.len = len(self.data)
        self.transform = Compose([
            ToTensor(),
            Resize((112, 112)),
            addGuassianNoise(mean=0,std=0.05)

        ])

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.data_path, str(self.data[idx]))
            img_file = os.listdir(img_path)
            names = []
            for img in img_file:
                names.append(os.path.join(img_path, img)) 
            imgs = [cv2.imread(name, cv2.IMREAD_COLOR) for name in names]
            
            imgs = imgs[8:]
            imgs = imgs[0:30] if len(imgs) > 30 else imgs + [imgs[-1]] * (30 - len(imgs))
            
            imgs = [self.transform(img) for img in imgs]
            imgs = torch.stack(tensors=imgs, dim=0)
        except:
            print(self.data[idx])
        return imgs, self.labels[idx]

    def __len__(self):
        return self.len


class addGuassianNoise():
    def __init__(self, mean = 0.0,std = 0.1):
        self.mean = mean
        self.std = std

    def __call__(self,img):
        noise = torch.randn_like(img) * self.std + self.mean
        noise_img = img + noise
        noise_img.clamp_(0,1)
        return noise_img
