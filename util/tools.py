import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import filters
from torchvision import transforms


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


def gabor_transform(image):
    # 将图像转换为张量
    tensor_image = transforms.ToTensor()(image)

    # 将张量转换为numpy数组
    np_image = tensor_image.numpy()

    # 获取图像的通道数
    num_channels = np_image.shape[0]

    # 创建一个空的数组来存储Gabor变换后的图像
    transformed_image = np.zeros_like(np_image)

    # 对每个通道应用Gabor变换
    for channel in range(num_channels):
        transformed_image[channel] = gabor_filter_cv2(np_image[channel])

    return torch.from_numpy(transformed_image)


def build_label_mapping(labels: list):
    """
    构建标签到数字的映射字典。

    参数:
    labels : list
        包含多个标签的列表。

    返回值:
    dict
        标签到数字的映射字典，其中每个标签对应一个唯一的数字。
    """
    label_mapping = {}
    # 遍历标签列表，为每个标签分配一个唯一的数字
    for num, label in enumerate(labels):
        label_mapping[label] = num
    return label_mapping
