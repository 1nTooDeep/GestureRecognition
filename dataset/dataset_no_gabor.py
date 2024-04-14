import os.path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
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
            Resize((112, 112))

        ])
        print(f"Size of GestureDataSet is {self.len}.")

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, str(self.data[idx]))
        img_file = os.listdir(img_path)
        names = []
        for img in img_file:
            names.append(os.path.join(img_path, img))
        imgs = [cv2.imread(name, cv2.IMREAD_COLOR) for name in names]
        imgs = imgs[:30] if len(imgs) > 30 else imgs + [imgs[-1]] * (30 - len(imgs))
        
        imgs = [self.transform(img) for img in imgs]
        # print(imgs[0])
        # print(imgs[0].shape)
        imgs = torch.stack(tensors=imgs, dim=0)
        imgs = torch.transpose(imgs,0,1)
        return imgs, self.labels[idx]

    def __len__(self):
        return self.len
