import os.path

import cv2
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize,Lambda,CenterCrop
from torch.utils.data import DataLoader
from config import Config
import random
def frame_extract(input_frames):
        """
        Given a list of input frames, randomly selects 30 unique frames if the list length is greater than 30.
        If the list length is less than or equal to 30, some frames may be duplicated while preserving their ascending order.

        Args:
            input_frames (list): The list of input frames.

        Returns:
            list: A new list containing 30 frames according to the specified conditions.
        """

        # Copy and sort the input list in ascending order
        sorted_frames = sorted(input_frames.copy())

        # If the list length is greater than 30, randomly select 30 unique frames
        if len(sorted_frames) > 30:
            selected_frames = random.sample(sorted_frames, 30)
            return sorted(selected_frames)

        # If the list length is less than or equal to 30, duplicate frames as needed to reach a length of 30
        result = []
        for _ in range(30 - len(input_frames)):
            selected_frame = random.choice(sorted_frames)
            result.append(selected_frame)
        result = sorted(result + input_frames)
        return result
        

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
            CenterCrop((112, 112)),
            addGuassianNoise(mean=0,std=0.1)

        ])

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.data_path, str(self.data[idx]))
            img_file = os.listdir(img_path)
            img_file = sorted(img_file)
            img_file = frame_extract(img_file)
            names = []
            for img in img_file:
                names.append(os.path.join(img_path, img)) 
            imgs = [cv2.imread(name, cv2.IMREAD_COLOR) for name in names]
            imgs = [self.transform(img) for img in imgs]
            imgs = torch.stack(tensors=imgs, dim=0)
        except:
            print(self.data[idx])
            exit()
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
