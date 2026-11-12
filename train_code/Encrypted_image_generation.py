import copy
import os
import math
import time
import torch
import hashlib
import warnings
import matplotlib
import numpy as np
import config as c
from utils.jpeg import *
from pytorch_msssim import ssim
from loss_calculation import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Image_attack import image_attack
from model_inn.encoder_decoder import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# ================================<<环境设备设置>>================================ #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')


# ================================<<环境设备设置>>================================ #

def load(model, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)
    tem_step = state_dicts['tem_step']
    return model, tem_step


def hash_generation(tensor_images):
    tensor_images = tensor_images.cpu()
    byte_images = tensor_images.numpy().tobytes()
    sha256_hash = hashlib.sha256(byte_images).hexdigest()
    # 将十六进制哈希值转换为二进制比特
    binary_bits = bin(int(sha256_hash, 16))[2:].zfill(256)
    return binary_bits


def Random_bits_generation(length, seed=None):
    if seed is not None:
        random.seed(seed)
    return ''.join(random.choice('01') for _ in range(length))


def xor_binary_bits(bits1, bits2):
    return ''.join('1' if b1 != b2 else '0' for b1, b2 in zip(bits1, bits2))


def key_generation(img, key2):
    key1_hash = hash_generation(img)
    key2_bits = Random_bits_generation(len(key1_hash), key2)
    key = xor_binary_bits(key1_hash, key2_bits)
    decimal_value = int(key, 2)
    return decimal_value


def set_large_seed(seed):
    # Convert the seed to a string and encode it
    seed_str = str(seed).encode('utf-8')
    sha256 = hashlib.sha256(seed_str).hexdigest()
    hash_int = int(sha256, 16)
    torch_seed = hash_int % (2 ** 32)
    torch.manual_seed(torch_seed)
    print(f"Seed set to: {torch_seed}")


if not os.path.exists(c.original_path):
    os.makedirs(c.original_path)

if not os.path.exists(c.cor_encryption_path):
    os.makedirs(c.cor_encryption_path)

if not os.path.exists(c.err1_encryption_path):
    os.makedirs(c.err1_encryption_path)

if not os.path.exists(c.err2_encryption_path):
    os.makedirs(c.err2_encryption_path)

# ==========================<< Dataset load >>========================== #
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
test_dataset = ImageFolder(root=c.test_data_path, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
# ==========================<< Model load >>========================== #
model = INL()
model.to(device)
model, tem_step = load(model, './save_model/INL_30000.pth')
model.eval()
with torch.no_grad():
    idx = 0
    for original_img, _ in test_loader:
        idx += 1
        original_img = original_img.to(device)
        # ===========================<<Key generation>>=========================== #
        cor_sub_key1 = 26
        cor_key = key_generation(original_img, cor_sub_key1)
        modified_img = copy.deepcopy(original_img)
        modified_img[0, 1, 1, 1] = modified_img[0, 1, 1, 1] + 1 / 255
        err_key1 = key_generation(modified_img, cor_sub_key1)
        cor_sub_key2 = 27
        err_key2 = key_generation(original_img, cor_sub_key2)

        cor_key = cor_key % (2 ** 32)
        torch.manual_seed(cor_key)
        ran_img_cor = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
        ran_img_cor = ran_img_cor.repeat(original_img.shape[0], 1, 1, 1)
        ran_img_cor = ran_img_cor.to(device)

        err_key1 = err_key1 % (2 ** 32)
        torch.manual_seed(err_key1)
        ran_img_err1 = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
        ran_img_err1 = ran_img_err1.repeat(original_img.shape[0], 1, 1, 1)
        ran_img_err1 = ran_img_err1.to(device)

        err_key2 = err_key2 % (2 ** 32)
        torch.manual_seed(err_key2)
        ran_img_err2 = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
        ran_img_err2 = ran_img_err2.repeat(original_img.shape[0], 1, 1, 1)
        ran_img_err2 = ran_img_err2.to(device)

        # ============<<Image encryption>>============ #
        input_data_cor = [ran_img_cor, original_img]
        input_data_err1 = [ran_img_err1, modified_img]
        input_data_err2 = [ran_img_err2, original_img]
        _, stego_img_cor, _, _ = model(input_data_cor)
        _, stego_img_err1, _, _ = model(input_data_err1)
        _, stego_img_err2, _, _ = model(input_data_err2)

        stego_img_cor = torch.clamp(stego_img_cor, 0, 1)
        stego_img_err1 = torch.clamp(stego_img_err1, 0, 1)
        stego_img_err2 = torch.clamp(stego_img_err2, 0, 1)

        original_img = transforms.ToPILImage()(original_img.squeeze())
        image_path = os.path.join(c.original_path, f"{idx}.png")
        original_img.save(image_path)

        stego_img_cor = transforms.ToPILImage()(stego_img_cor.squeeze())
        image_path = os.path.join(c.cor_encryption_path, f"{idx}.png")
        stego_img_cor.save(image_path)

        stego_img_err1 = transforms.ToPILImage()(stego_img_err1.squeeze())
        image_path = os.path.join(c.err1_encryption_path, f"{idx}.png")
        stego_img_err1.save(image_path)

        stego_img_err2 = transforms.ToPILImage()(stego_img_err2.squeeze())
        image_path = os.path.join(c.err2_encryption_path, f"{idx}.png")
        stego_img_err2.save(image_path)
