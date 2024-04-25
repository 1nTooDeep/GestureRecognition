import os.path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import *
from config import Config

def gabor_filter_cv2(image):
    ksize = (31, 31)
    sigma = 2.0
    theta = [0, 1, 2, 3]
    lambd = 0.4
    gamma = [1, 2, 3, 4, 5, 6]
    imgs = []
    # 创建Gabor滤波器
    for t in theta:
        for g in gamma:
            kernel = cv2.getGaborKernel(ksize, sigma, t, lambd, g, 0, ktype=cv2.CV_32F)
            # 对图像应用Gabor滤波器
            filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            imgs.append(filtered_image)
    return imgs


class GestureDataSet(Dataset):
    def __init__(self, data_path: str, data: pd.DataFrame):
        super(GestureDataSet).__init__()
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
        try:
            img_file = os.listdir(img_path)
            names = []
            for img in img_file:
                names.append(os.path.join(img_path, img))
            imgs = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in names]
            # imgs = imgs[:30] if len(imgs) > 30 else imgs + [imgs[-1]] * (30 - len(imgs))
            # zoom img to 112 * 112
            # imgs = [cv2.resize(img, (112, 112)) for img in imgs]
            # imgs = [(img.astype(np.float32) - self.mean) / self.std for img in imgs]
            # imgs = [gabor_filter_cv2(img) for img in imgs]
            # imgs = [torch.from_numpy(np.array(img)) for img in imgs]
            # data = torch.stack(tensors=imgs, dim=0)
            # data = torch.transpose(data, 0, 1)
            return imgs, self.labels[idx]
        except Exception:
            raise Exception("No images found in {}".format(img_path))

    def __len__(self):
        return self.len

