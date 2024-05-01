import os.path
import random

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, CenterCrop,Normalize

from config import Config


def frame_extract(input_frames,num_frames):
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
    if len(sorted_frames) > num_frames:
        selected_frames = random.sample(sorted_frames, num_frames)
        return sorted(selected_frames)

    # If the list length is less than or equal to 30, duplicate frames as needed to reach a length of 30
    result = []
    for _ in range(num_frames - len(input_frames)):
        selected_frame = random.choice(sorted_frames)
        result.append(selected_frame)
    result = sorted(result + input_frames)
    return result

class GestureDataSet(Dataset):
    def __init__(self, data_path: str, data: pd.DataFrame,num_frames = 30,mode = 'train',**kwargs):
        super(GestureDataSet).__init__()
        self.config = Config()
        self.data_path = data_path
        self.data = data['num']
        self.labels = data['label']
        self.num_frames = num_frames
        self.len = len(self.data)
        self.mean = [ 0.4377,  0.4047,  0.3925]
        self.std = [0.2674,  0.2676,  0.2648]
        if mode == 'train':
            self.transform = Compose([
                ToTensor(),
                CenterCrop((112, 112)),
                Normalize(mean=self.mean, std=self.std),
                GaussNoise(mean=0.0, std=0.1),
            ])
        else:
            self.transform = Compose([
                ToTensor(),
                CenterCrop((112, 112)),
                Normalize(mean=self.mean, std=self.std),
            ])
        self.first = kwargs.get("first") if kwargs.get("first") is not None else "frame"

    def __getitem__(self, idx):
        try:
            # get images path
            img_path = os.path.join(self.data_path, str(self.data[idx]))
            img_file = os.listdir(img_path)
            # sort images into frames order
            img_file = sorted(img_file)
            # extract frames
            img_file = frame_extract(img_file,self.num_frames)
            names = []
            for img in img_file:
                names.append(os.path.join(img_path, img))
            imgs = [cv2.imread(name, cv2.IMREAD_COLOR) for name in names]
            imgs = [self.transform(img) for img in imgs]
            imgs = torch.stack(tensors=imgs, dim=0)
            if self.first == "channel":
                imgs = imgs.transpose(0, 1)
        except Exception as e:
            print(self.data[idx],e)
            exit(1)
        return imgs, self.labels[idx]

    def __len__(self):
        return self.len

class GaussNoise():
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
    def __call__(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        return img + noise