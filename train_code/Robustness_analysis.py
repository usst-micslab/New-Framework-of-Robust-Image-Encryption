import os
import math
import time
import torch
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


# ==========================<< Dataset load >>========================== #
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
test_dataset = ImageFolder(root=c.test_data_path, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=5, shuffle=False)
# ==========================<< Model load >>========================== #


model = INL()
model.to(device)
model, tem_step = load(model, './save_model/INL_92000.pth')
# Unattack， JPEG_compression_atttack，Gaussian_noise_attack
# Scaling_attack，Rotation_attack，Center_attack
attack_method = 10
# QF_list = [90, 80, 70, 60, 50]
# QF_list = [0.01, 0.02, 0.03, 0.04, 0.05]
# QF_list = [0.9, 0.8, 0.7, 0.6, 0.5]
# QF_list = [1, 2, 3, 4, 5]
# QF_list = [0.06, 0.07, 0.08, 0.09, 0.1]
QF_list = [0.8, 0.9, 1.1, 1.2]
model.eval()
with torch.no_grad():
    for QF in QF_list:
        start = time.time()
        idx = 0
        psnr_en_sum, psnr_cor_sum, psnr_corwn_sum = 0., 0., 0.
        ssim_en_sum, ssim_cor_sum, ssim_corwn_sum = 0., 0., 0.
        for original_img, _ in test_loader:
            idx += 1
            original_img = original_img.to(device)
            # ===========================<<Key generation>>=========================== #
            torch.manual_seed(26)
            ran_img_cor = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
            ran_img_cor = ran_img_cor.repeat(original_img.shape[0], 1, 1, 1)
            ran_img_cor = ran_img_cor.to(device)

            # ============<<Image encryption>>============ #
            input_data_cor = [ran_img_cor, original_img]
            _, stego_img, _, _ = model(input_data_cor)
            stego_img = torch.clamp(stego_img, 0, 1)

            # ====================<<攻击方法>>==================== #
            stego_img_w_n = image_attack(attack_method, stego_img, QF)

            # ============<<Image decryption>>============ #
            output_data_cor = [ran_img_cor, stego_img]
            output_data_cor_w_n = [ran_img_cor, stego_img_w_n]

            _, de_img_cor = model(output_data_cor, rev=True)
            _, de_img_cor_w_n = model(output_data_cor_w_n, rev=True)

            de_img_cor = torch.clamp(de_img_cor, 0, 1)
            de_img_cor_w_n = torch.clamp(de_img_cor_w_n, 0, 1)
            # =========================<<PSNR calculation>>========================= #
            psnr_en = psnr(stego_img, original_img)
            psnr_de_cor = psnr(de_img_cor, original_img)
            psnr_de_cor_w_n = psnr(de_img_cor_w_n, original_img)

            ssim_en = ssim(stego_img, original_img, data_range=1.0)
            ssim_de_cor = ssim(de_img_cor, original_img, data_range=1.0)
            ssim_de_cor_w_n = ssim(de_img_cor_w_n, original_img, data_range=1.0)

            psnr_en_sum += psnr_en
            psnr_cor_sum += psnr_de_cor
            psnr_corwn_sum += psnr_de_cor_w_n

            ssim_en_sum += ssim_en
            ssim_cor_sum += ssim_de_cor
            ssim_corwn_sum += ssim_de_cor_w_n

        end = time.time()
        print(f'The QF is : {QF}')
        print(
            f'The mean psnr: psnr_en:{psnr_en_sum / idx:.2f}, | psnr_de_cor:{psnr_cor_sum / idx:.2f}, |, psnr_de_cor_w_n:{psnr_corwn_sum / idx:.2f}')
        print(
            f'The mean ssim: ssim_en:{ssim_en_sum / idx:.4f}, | ssim_de_cor:{ssim_cor_sum / idx:.4f}, |, ssim_de_cor_w_n:{ssim_corwn_sum / idx:.4f}')
        print('Encryption and decryption time for 1000 images: ', end - start)






