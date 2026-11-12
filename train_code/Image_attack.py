import os
import math
import torch
import warnings
import matplotlib
import numpy as np
import config as c
from utils.jpeg import *
from loss_calculation import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model_inn.encoder_decoder import *
from utils.jpeg import JpegSS, JpegTest

# ================================<<环境设备设置>>================================ #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')


# ================================<<环境设备设置>>================================ #

class BatchMedianFilter:
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, batch):
        return self.batch_median_filter(batch, self.kernel_size)

    def batch_median_filter(self, batch, kernel_size):
        batch_size, channels, height, width = batch.size()
        pad = kernel_size // 2

        # Padding the batch
        batch_padded = F.pad(batch, (pad, pad, pad, pad), mode='reflect')

        # Unfold the batch
        batch_unfolded = batch_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)

        # Reshape to (batch_size, channels, height, width, kernel_size*kernel_size)
        batch_unfolded = batch_unfolded.contiguous().view(batch_size, channels, height, width, -1)

        # Compute the median along the last dimension
        batch_median = batch_unfolded.median(dim=-1)[0]

        return batch_median


class HighPassFilter(nn.Module):
    def __init__(self):
        super(HighPassFilter, self).__init__()
        # 定义高通滤波卷积核
        self.high_pass_filter = torch.tensor([[-1, -1, -1],
                                              [-1, 8, -1],
                                              [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # 扩展卷积核以匹配输入图像的通道数
        filter = self.high_pass_filter.expand(channels, 1, 3, 3).to(x.device)
        # 对批次图像应用高通滤波
        x = F.conv2d(x, filter, padding=1, groups=channels)
        return x


class MeanFilter(nn.Module):
    def __init__(self, kernel_size=3):
        super(MeanFilter, self).__init__()
        # 定义均值滤波卷积核
        self.kernel_size = kernel_size
        self.mean_filter = torch.ones((kernel_size, kernel_size), dtype=torch.float32) / (kernel_size * kernel_size)
        self.mean_filter = self.mean_filter.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # 扩展卷积核以匹配输入图像的通道数
        filter = self.mean_filter.expand(channels, 1, self.kernel_size, self.kernel_size).to(x.device)
        # 对批次图像应用均值滤波
        x = F.conv2d(x, filter, padding=self.kernel_size // 2, groups=channels)
        return x


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=7, sigma=1.0):
        super(GaussianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_filter = self.create_gaussian_kernel(kernel_size, sigma)
        self.gaussian_filter = self.gaussian_filter.unsqueeze(0).unsqueeze(0)

    def create_gaussian_kernel(self, kernel_size, sigma):
        k = kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        y = torch.arange(-k, k + 1, dtype=torch.float32)
        x, y = torch.meshgrid(x, y)
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        filter = self.gaussian_filter.expand(channels, 1, self.kernel_size, self.kernel_size).to(x.device)
        x = F.conv2d(x, filter, padding=self.kernel_size // 2, groups=channels)
        return x


def add_gaussian_noise(image_tensor, mean=0, std=0.1):
    """
    将高斯噪声添加到图像张量中。

    Args:
    - image_tensor: 输入的图像张量，形状为 (batch_size, channels, height, width)
    - mean: 高斯分布的均值，默认为0
    - std: 高斯分布的标准差（噪声强度），默认为0.1

    Returns:
    - noisy_image_tensor: 添加噪声后的图像张量，形状与输入相同
    """

    # 生成与输入张量相同形状的高斯噪声张量
    noise = torch.randn_like(image_tensor) * std + mean

    # 将高斯噪声添加到输入图像张量中
    noisy_image_tensor = image_tensor + noise

    return noisy_image_tensor


def add_salt_and_pepper_noise(image_tensor, salt_prob=0.05, pepper_prob=0.05):
    """
    向图像Tensor添加椒盐噪声
    :param image_tensor: 输入图像的Tensor
    :param salt_prob: 椒盐噪声中盐的概率，默认为0.05
    :param pepper_prob: 椒盐噪声中椒的概率，默认为0.05
    :return: 添加了椒盐噪声的图像Tensor
    """
    # 生成与图像张量相同大小的随机数张量
    salt_pepper_noise = torch.rand(image_tensor.shape)

    # 将符合盐和椒噪声概率条件的像素值设为255（白色）和0（黑色）
    image_tensor[salt_pepper_noise < salt_prob] = 1.0  # 盐噪声
    image_tensor[salt_pepper_noise > (1 - pepper_prob)] = 0.0  # 椒噪声

    return image_tensor


def mask_image(input, tamp_rate):
    mask = torch.ones(input.size(2), input.size(3)).to(input.device)  # 在 [0, 1] 范围内生成随机张量
    total_num = mask.size(0) * mask.size(1)
    tamp_num = int(tamp_rate * total_num)
    zero_indices = random.sample(range(total_num), tamp_num)
    zero_indices.sort()
    for idx in zero_indices:
        row = idx // mask.size(0)
        col = idx % mask.size(1)
        mask[row, col] = 0

    tamp_img = input * mask
    return tamp_img


def center_scale_to_zero(image_tensor, scale=0.5):
    """
    将图像的中心按比例设置为0
    :param image_tensor: 输入图像的Tensor，形状为[C, H, W]
    :param scale: 中心像素设置为0的比例，默认为0.5
    :return: 处理后的图像Tensor
    """
    # 计算要设置为0的像素数量
    num_pixels_to_set_zero = int(scale * min(image_tensor.size(2), image_tensor.size(3)))

    # 中心区域的高度和宽度
    center_height = image_tensor.size(2) // 2
    center_width = image_tensor.size(3) // 2

    # 创建一个和image_tensor相同大小的全0张量
    result_tensor = torch.zeros_like(image_tensor)

    # 计算矩形的位置
    start_h = center_height - num_pixels_to_set_zero
    end_h = center_height + num_pixels_to_set_zero
    start_w = center_width - num_pixels_to_set_zero
    end_w = center_width + num_pixels_to_set_zero

    # 将中心区域的像素值设置为原图像对应区域的值
    # result_tensor[:, :, start_h:end_h, start_w:end_w] = image_tensor[:, :, start_h:end_h, start_w:end_w]
    image_tensor[:, :, start_h:end_h, start_w:end_w] = 0

    return image_tensor


def image_attack(attack_method, stego_img, QF):
    if attack_method == 1:
        # JPEG_attack
        JPEG_attack = JpegTest(QF)
        stego_img_w_n = JPEG_attack(stego_img.clone())

    elif attack_method == 2:
        # adding guass_noise
        stego_img_w_n = add_gaussian_noise(stego_img.clone(), std=QF)
        # stego_img_w_n = add_salt_and_pepper_noise(stego_img.clone(), salt_prob=0.01, pepper_prob=0.01)

    elif attack_method == 3:
        # scaling attack
        noised_image = F.interpolate(stego_img.clone(),
                                     size=(int(stego_img.shape[2] * QF), int(stego_img.shape[3] * QF)),
                                     scale_factor=None,
                                     mode='bilinear', align_corners=False)
        stego_img_w_n = F.interpolate(noised_image, size=(stego_img.shape[2], stego_img.shape[3]), scale_factor=None,
                                      mode='bilinear',
                                      align_corners=False)

    elif attack_method == 4:
        # rotation attack
        stego_img_w_n = TF.rotate(stego_img.cpu(), angle=QF).to(device)

    elif attack_method == 5:
        # crop attack
        # stego_img_w_n = mask_image(stego_img.clone(), 0.05)
        stego_img_w_n = center_scale_to_zero(stego_img.clone(), scale=QF)

    elif attack_method == 6:
        # Create the Median filter
        median_filter = BatchMedianFilter(kernel_size=QF)
        stego_img_w_n = median_filter(stego_img.clone())

    elif attack_method == 7:
        # 创建高通滤波器实例
        high_pass_filter = HighPassFilter()
        stego_img_w_n = high_pass_filter(stego_img.clone())

    elif attack_method == 8:
        # 创建均值滤波器实例
        mean_filter = MeanFilter(kernel_size=QF)
        stego_img_w_n = mean_filter(stego_img.clone())

    elif attack_method == 9:
        # 创建高斯滤波器实例
        gaussian_filter = GaussianFilter(kernel_size=QF, sigma=1.0)
        stego_img_w_n = gaussian_filter(stego_img.clone())

    elif attack_method == 10:
        # Gamma correction attack
        gamma = QF
        stego_img_w_n = torch.pow(stego_img.clone(), gamma)

    return stego_img_w_n
