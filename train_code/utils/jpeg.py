import os
import random
import string
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from utils.utils import clamp
import torchvision.transforms.functional as TF


class JpegTest(nn.Module):
    def __init__(self, Q, subsample=2, path="temp/"):
        super(JpegTest, self).__init__()
        self.Q = Q
        self.subsample = subsample
        self.path = path
        if not os.path.exists(path): os.mkdir(path)

    def get_path(self):
        return self.path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".jpg"

    def forward(self, image):
        shape = image.shape
        noised_image = torch.zeros_like(image)
        for i in range(shape[0]):
            single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)

            file = self.get_path()
            while os.path.exists(file):
                file = self.get_path()
            im.save(file, format="JPEG", quality=self.Q, subsampling=self.subsample)
            jpeg = np.array(Image.open(file), dtype=np.uint8)
            os.remove(file)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            noised_image[i] = transform(jpeg).unsqueeze(0).to(image.device)

        return noised_image


class JpegBasic(nn.Module):
    def __init__(self):
        super(JpegBasic, self).__init__()

    def std_quantization(self, image_yuv_dct, scale_factor, round_func=torch.round):

        luminance_quant_tbl = (torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
            image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

        chrominance_quant_tbl = (torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
            image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

        q_image_yuv_dct = image_yuv_dct.clone()
        q_image_yuv_dct[:, :1, :, :] = image_yuv_dct[:, :1, :, :] / luminance_quant_tbl
        q_image_yuv_dct[:, 1:, :, :] = image_yuv_dct[:, 1:, :, :] / chrominance_quant_tbl
        q_image_yuv_dct_round = round_func(q_image_yuv_dct)
        return q_image_yuv_dct_round

    def std_reverse_quantization(self, q_image_yuv_dct, scale_factor):

        luminance_quant_tbl = (torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
            q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

        chrominance_quant_tbl = (torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
            q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

        image_yuv_dct = q_image_yuv_dct.clone()
        image_yuv_dct[:, :1, :, :] = q_image_yuv_dct[:, :1, :, :] * luminance_quant_tbl
        image_yuv_dct[:, 1:, :, :] = q_image_yuv_dct[:, 1:, :, :] * chrominance_quant_tbl
        return image_yuv_dct

    def dct(self, image):
        coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image.shape[2] // 8
        image_dct = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
        image_dct = torch.matmul(coff, image_dct)
        image_dct = torch.matmul(image_dct, coff.permute(1, 0))
        image_dct = torch.cat(torch.cat(image_dct.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image_dct

    def idct(self, image_dct):
        coff = torch.zeros((8, 8), dtype=torch.float).to(image_dct.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image_dct.shape[2] // 8
        image = torch.cat(torch.cat(image_dct.split(8, 2), 0).split(8, 3), 0)
        image = torch.matmul(coff.permute(1, 0), image)
        image = torch.matmul(image, coff)
        image = torch.cat(torch.cat(image.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image

    def rgb2yuv(self, image_rgb):
        image_yuv = torch.empty_like(image_rgb)
        image_yuv[:, 0:1, :, :] = 0.299 * image_rgb[:, 0:1, :, :] \
                                  + 0.587 * image_rgb[:, 1:2, :, :] + 0.114 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 1:2, :, :] = -0.1687 * image_rgb[:, 0:1, :, :] \
                                  - 0.3313 * image_rgb[:, 1:2, :, :] + 0.5 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 2:3, :, :] = 0.5 * image_rgb[:, 0:1, :, :] \
                                  - 0.4187 * image_rgb[:, 1:2, :, :] - 0.0813 * image_rgb[:, 2:3, :, :]
        return image_yuv

    def yuv2rgb(self, image_yuv):
        image_rgb = torch.empty_like(image_yuv)
        image_rgb[:, 0:1, :, :] = image_yuv[:, 0:1, :, :] + 1.40198758 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 1:2, :, :] = image_yuv[:, 0:1, :, :] - 0.344113281 * image_yuv[:, 1:2, :, :] \
                                  - 0.714103821 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 2:3, :, :] = image_yuv[:, 0:1, :, :] + 1.77197812 * image_yuv[:, 1:2, :, :]
        return image_rgb

    def yuv_dct(self, image, subsample):
        image = (clamp(image, -1, 1) + 1) * 255 / 2

        pad_height = (8 - image.shape[2] % 8) % 8
        pad_width = (8 - image.shape[3] % 8) % 8
        image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(image)

        image_yuv = self.rgb2yuv(image)

        assert image_yuv.shape[2] % 8 == 0
        assert image_yuv.shape[3] % 8 == 0

        image_subsample = self.subsampling(image_yuv, subsample)

        image_dct = self.dct(image_subsample)

        return image_dct, pad_width, pad_height

    def idct_rgb(self, image_quantization, pad_width, pad_height):
        image_idct = self.idct(image_quantization)

        image_ret_padded = self.yuv2rgb(image_idct)

        image_rgb = image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height,
                    :image_ret_padded.shape[3] - pad_width].clone()

        image_rgb = image_rgb * 2 / 255 - 1
        return clamp(image_rgb, -1, 1)

    def subsampling(self, image, subsample):
        if subsample == 2:
            split_num = image.shape[2] // 8
            image_block = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
            for i in range(8):
                if i % 2 == 1: image_block[:, 1:3, i, :] = image_block[:, 1:3, i - 1, :]
            for j in range(8):
                if j % 2 == 1: image_block[:, 1:3, :, j] = image_block[:, 1:3, :, j - 1]
            image = torch.cat(torch.cat(image_block.chunk(split_num, 0), 3).chunk(split_num, 0), 2)
        return image


class Jpeg(JpegBasic):
    def __init__(self, Q, subsample=0):
        super(Jpeg, self).__init__()

        self.Q = Q
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

        self.subsample = subsample

    def forward(self, image):
        image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

        image_quantization = self.std_quantization(image_dct, self.scale_factor)

        image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

        noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
        return noised_image


class JpegSS(JpegBasic):
    def __init__(self, QF, noise_str, scale_str, subsample=0):
        super(JpegSS, self).__init__()

        self.QF = QF
        self.noise_str = noise_str
        self.scale_str = scale_str
        self.scale_factor = 2 - self.QF * 0.02 if self.QF >= 50 else 50 / self.QF

        self.subsample = subsample

    def round_ss(self, x):
        cond = torch.tensor((torch.abs(x) < 0.5), dtype=torch.float).to(x.device)
        return cond * (x ** 3) + (1 - cond) * x

    def mask_image(self, input, tamp_rate=0.3):
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

    def random_modify(self, tensor):
        """
        随机篡改一个 tensor 中的一个区域为 0.

        参数:
        tensor: 需要篡改的 4D tensor，形状为 (batch_size, channels, height, width)
        patch_size: 需要篡改区域的大小，默认为 25

        返回:
        modified_tensor: 篡改后的 tensor
        """
        # 确保 tensor 的形状是 (batch_size, channels, height, width)
        assert tensor.dim() == 4, "tensor 需要是一个 4D tensor"
        batch_size, channels, height, width = tensor.shape
        patch_size = int(height * 0.1)
        # 随机选择篡改区域的左上角位置
        top_left_x = random.randint(0, height - patch_size)
        top_left_y = random.randint(0, width - patch_size)

        # 创建一个与 tensor 相同大小的 mask，并初始化为 1
        mask = torch.ones_like(tensor)

        # 设置所有图像中相同位置的 mask 为 0
        mask[:, :, top_left_x:top_left_x + patch_size, top_left_y:top_left_y + patch_size] = 0

        # 将 mask 与 tensor 相乘
        modified_tensor = tensor * mask

        return modified_tensor

    def forward(self, image):
        # ========= <<JPEG attack simulation>> ========= #
        image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)
        image_quantization = self.std_quantization(image_dct, self.scale_factor, self.round_ss)
        image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)
        noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)

        # # ========= <<Gaussian noise attack>> ========= #
        # mean_value = 0
        # noise = torch.randn_like(noised_image) * self.noise_str + mean_value
        # noised_image = noised_image + noise
        #
        # # ========= <<scaling attack>> ========= #
        # noised_image = F.interpolate(noised_image,
        #                              size=(int(image.shape[2] * self.scale_str), int(image.shape[3] * self.scale_str)),
        #                              scale_factor=None, mode='bilinear', align_corners=False)
        # noised_image = F.interpolate(noised_image, size=(image.shape[2], image.shape[3]),
        #                              scale_factor=None, mode='bilinear', align_corners=False)

        # ========= <<rotation attack>> ========= #
        # random_angle = torch.randint(0, 31, (1,)).item()
        # noised_image = TF.rotate(noised_image.cpu(), angle=random_angle).to(image.device)
        #
        # ========= <<Tampering attack>> ========= #
        # noised_image = self.mask_image(noised_image)
        # noised_image = self.random_modify(noised_image)
        return noised_image


class DCT_and_IDCT(JpegBasic):
    def __init__(self, subsample=0, pad_width=0, pad_height=0):
        self.subsample = subsample
        self.pad_width = pad_width
        self.pad_height = pad_height

    def img_dct(self, image):
        image_dct = self.dct(image)
        return image_dct

    def img_idct(self, image_dct):
        out_image = self.idct(image_dct)
        return out_image
