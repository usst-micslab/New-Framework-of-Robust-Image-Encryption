import numpy as np
import torch
# from utils.datasets import *
import torch.nn.functional as F
from torch import nn


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0 if img1.max() <= 1 else 255.0
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr_value


def MSE_loss_fn(input_img1, input_img2):
    loss_fn = torch.nn.MSELoss(reduce=True)
    loss = loss_fn(input_img1, input_img2)
    return loss


def compute_image_entropy(images):
    # images: Tensor of shape (batch_size, channels, height, width)
    # Convert tensor to numpy array and reshape
    images_np = images.cpu().detach().numpy().transpose((0, 2, 3, 1))
    images_np = np.clip(images_np * 255, 0, 255).astype(np.uint8)  # Convert to uint8

    batch_entropy = []
    for img in images_np:
        # Convert RGB image to grayscale
        gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        # Compute histogram
        hist, _ = np.histogram(gray_img.flatten(), bins=256, range=[0, 256])
        # Normalize histogram to obtain probability distribution
        prob_dist = hist / np.sum(hist)
        # Compute entropy
        entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))  # Add a small epsilon to avoid log(0)
        batch_entropy.append(entropy)

    return torch.tensor(batch_entropy)


def entropy_loss(images):
    entropy = compute_image_entropy(images)
    avg_entropy = entropy.mean()
    return avg_entropy


# def ssim(img1, img2, window_size=11, size_average=True):
#     img1 = img1.cpu() * 1
#     img2 = img2.cpu() * 1
#     window = create_window(window_size, img1.size(1))
#
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img1.size(1))
#
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2
#
#     C1 = (0.01 * 1) ** 2  # 通常为0.01
#     C2 = (0.03 * 1) ** 2  # 通常为0.03
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#
#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)
#
#
# def create_window(window_size, channel):
#     sigma = 1.5
#     gauss = torch.Tensor(
#         [torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))) for x in range(window_size)])
#     window = torch.outer(gauss, gauss)
#     window = window.view(1, 1, window_size, window_size)
#     window = window.expand(channel, 1, window_size, window_size).contiguous()
#     return window / window.sum()
#
#
# def gaussian(window_size, sigma):
#     gauss = torch.Tensor(
#         [torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))) for x in range(window_size)])
#     return gauss / gauss.sum()


class LpipsTorch(nn.Module):
    def __init__(self):
        super(LpipsTorch, self).__init__()
        from lpips_model import dist_model

        self.model = dist_model.DistModel()

    def load_model(self):
        self.model.initialize(use_gpu=True, version=0.1)
        for p in self.model.parameters:
            p.requires_grad = False

    def lpips(self, input0, input1):
        input0 = input0 * 2.0 - 1.0
        input1 = input1 * 2.0 - 1.0
        distance = self.model.net(input0, input1)
        distance = torch.squeeze(distance)
        # reshape the leading dimensions
        distance = torch.reshape(distance, [-1])
        return distance


def tensor_norm(tensor):
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    normalized_tensor = (tensor - min_value) / (max_value - min_value)
    return normalized_tensor, min_value, max_value


def tensor_inorm(tensor, min_value, max_value):
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def ran_img_generation(original_img, device):
    key_seed = torch.randint(0, 1000000000, (1,))
    torch.manual_seed(key_seed)
    ran_img_cor = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
    ran_img_cor = ran_img_cor.repeat(original_img.shape[0], 1, 1, 1)
    ran_img_cor = ran_img_cor.to(device)
    threshold = 0.5
    cor_key = (ran_img_cor >= threshold).float()
    cor_key = cor_key.to(device)

    key_seed_err = torch.randint(0, 1000000000, (1,))
    while key_seed == key_seed_err:
        key_seed_err = torch.randint(0, 1000000000, (1,))
    torch.manual_seed(key_seed_err)
    ran_img_err = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
    ran_img_err = ran_img_err.repeat(original_img.shape[0], 1, 1, 1)
    ran_img_err = ran_img_err.to(device)
    err_key = (ran_img_err >= threshold).float()
    err_key = err_key.to(device)

    target_key_seed = torch.randint(0, 1000000000, (1,))
    while key_seed == target_key_seed and key_seed_err == target_key_seed:
        target_key_seed = torch.randint(0, 1000000000, (1,))
    torch.manual_seed(target_key_seed)
    ran_tar_img = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
    ran_tar_img = ran_tar_img.repeat(original_img.shape[0], 1, 1, 1)
    ran_tar_img = ran_tar_img.to(device)
    tar_key = (ran_tar_img >= threshold).float()
    tar_key = tar_key.to(device)

    irr_key_seed = torch.randint(0, 1000000000, (1,))
    while key_seed == irr_key_seed and key_seed_err == irr_key_seed and target_key_seed == irr_key_seed:
        irr_key_seed = torch.randint(0, 1000000000, (1,))
    torch.manual_seed(irr_key_seed)
    ran_irr_img = torch.rand(1, original_img.shape[1], original_img.shape[2], original_img.shape[3])
    ran_irr_img = ran_irr_img.repeat(original_img.shape[0], 1, 1, 1)
    ran_irr_img = ran_irr_img.to(device)
    irr_key = (ran_irr_img >= threshold).float()
    irr_key = irr_key.to(device)
    return ran_img_cor, ran_img_err, ran_tar_img, ran_irr_img
