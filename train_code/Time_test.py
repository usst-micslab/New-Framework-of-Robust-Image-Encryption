import os
import math
import time
import torch
import warnings
import matplotlib
import config as c
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
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
# ==========================<< Model load >>========================== #
model = INL()
model.to(device)
model, tem_step = load(model, './save_model/INL_29000.pth')
model.eval()
with torch.no_grad():
    start = time.time()
    for original_img, _ in test_loader:
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

        # ============<<Image decryption>>============ #
        output_data_cor = [ran_img_cor, stego_img]
        _, de_img_cor = model(output_data_cor, rev=True)
        de_img_cor = torch.clamp(de_img_cor, 0, 1)

    end = time.time()
    print('Encryption and decryption time for 1000 images: ', end - start)
