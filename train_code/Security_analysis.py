import os
import cv2
import torch
import warnings
import numpy as np
import config as c
import pandas as pd
from tqdm import tqdm

# ================================<<环境设备设置>>================================ #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")


# ================================<<环境设备设置>>================================ #
def calculate_entropy(channel):
    hist, _ = np.histogram(channel, bins=256, range=(0, 256), density=True)
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    return entropy


def entropy(img):
    img = cv2.imread(img)
    w, h, _ = img.shape
    B, G, R = cv2.split(img)
    B_entropy = calculate_entropy(B)
    G_entropy = calculate_entropy(G)
    R_entropy = calculate_entropy(R)
    return R_entropy, G_entropy, B_entropy


def RGB_correlation(channel, N):
    '''
    分别计算图像通道相邻像素的水平、垂直和对角线的相关系数并返回
    '''
    h, w = channel.shape
    row = np.random.randint(0, h - 1, N)
    col = np.random.randint(0, w - 1, N)
    x, h_y, v_y, d_y = [], [], [], []
    for i in range(N):
        x.append(channel[row[i]][col[i]])
        h_y.append(channel[row[i]][col[i] + 1])
        v_y.append(channel[row[i] + 1][col[i]])
        d_y.append(channel[row[i] + 1][col[i] + 1])
    x = x * 3
    y = h_y + v_y + d_y

    ex = np.mean([channel[row[i]][col[i]] for i in range(N)])
    dx = np.var([channel[row[i]][col[i]] for i in range(N)])

    def calc_rxy(ey, dy, cov):
        return cov / (np.sqrt(dx) * np.sqrt(dy))

    def calc_cov(y_list, ey):
        return np.mean([(channel[row[i]][col[i]] - ex) * (y - ey) for i, y in enumerate(y_list)])

    h_ey = np.mean(h_y)
    v_ey = np.mean(v_y)
    d_ey = np.mean(d_y)

    h_dy = np.var(h_y)
    v_dy = np.var(v_y)
    d_dy = np.var(d_y)

    h_cov = calc_cov(h_y, h_ey)
    v_cov = calc_cov(v_y, v_ey)
    d_cov = calc_cov(d_y, d_ey)

    return calc_rxy(h_ey, h_dy, h_cov), calc_rxy(v_ey, v_dy, v_cov), calc_rxy(d_ey, d_dy, d_cov), x, y


def correlation(img, N=5000):
    img = cv2.imread(img)
    B, G, R = cv2.split(img)
    return RGB_correlation(R, N)[0:3], RGB_correlation(G, N)[0:3], RGB_correlation(B, N)[0:3]


def NPCR(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    w, h, _ = img1.shape
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)

    def calc_npcr(c1, c2):
        ar, num = np.unique((c1 != c2), return_counts=True)
        return 0 if len(num) == 1 else (num[0] if ar[0] == True else num[1]) / (w * h)

    return calc_npcr(R1, R2), calc_npcr(G1, G2), calc_npcr(B1, B2)


def UACI(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    w, h, _ = img1.shape
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)

    def calc_uaci(c1, c2):
        return np.sum(np.abs(c1.astype(np.int16) - c2.astype(np.int16))) / 255 / (w * h)

    return calc_uaci(R1, R2), calc_uaci(G1, G2), calc_uaci(B1, B2)


def main():
    results = []
    for idx in tqdm(range(1, 1001), desc="Processing images"):
        original_img = os.path.join(c.original_path, f"{idx}.png")
        En_cor_img = os.path.join(c.cor_encryption_path, f"{idx}.png")
        En_err1_img = os.path.join(c.err1_encryption_path, f"{idx}.png")
        En_err2_img = os.path.join(c.err2_encryption_path, f"{idx}.png")

        # =============== <<信息熵>> =============== #
        R_entropy1, G_entropy1, B_entropy1 = entropy(original_img)
        R_entropy2, G_entropy2, B_entropy2 = entropy(En_cor_img)

        # =============== <<相关性系数>> =============== #
        R_Rxy1, G_Rxy1, B_Rxy1 = correlation(original_img)
        R_Rxy2, G_Rxy2, B_Rxy2 = correlation(En_cor_img)

        # ============= <<抗差分分析>> ============= #
        R_npcr1, G_npcr1, B_npcr1 = NPCR(En_cor_img, En_err1_img)
        R_uaci1, G_uaci1, B_uaci1 = UACI(En_cor_img, En_err1_img)

        # ============= <<密钥敏感性分析>> ============= #
        R_npcr2, G_npcr2, B_npcr2 = NPCR(En_cor_img, En_err2_img)
        R_uaci2, G_uaci2, B_uaci2 = UACI(En_cor_img, En_err2_img)

        results.append([
            idx,
            R_entropy1, G_entropy1, B_entropy1,
            R_entropy2, G_entropy2, B_entropy2,
            R_Rxy1[0], R_Rxy1[1], R_Rxy1[2],
            G_Rxy1[0], G_Rxy1[1], G_Rxy1[2],
            B_Rxy1[0], B_Rxy1[1], B_Rxy1[2],
            R_Rxy2[0], R_Rxy2[1], R_Rxy2[2],
            G_Rxy2[0], G_Rxy2[1], G_Rxy2[2],
            B_Rxy2[0], B_Rxy2[1], B_Rxy2[2],
            R_npcr1, G_npcr1, B_npcr1,
            R_uaci1, G_uaci1, B_uaci1,
            R_npcr2, G_npcr2, B_npcr2,
            R_uaci2, G_uaci2, B_uaci2
        ])

    df = pd.DataFrame(results, columns=[
        'Index',
        'R_entropy1', 'G_entropy1', 'B_entropy1',
        'R_entropy2', 'G_entropy2', 'B_entropy2',
        'R_Rxy1_Horizontal', 'R_Rxy1_Vertical', 'R_Rxy1_Diagonal',
        'G_Rxy1_Horizontal', 'G_Rxy1_Vertical', 'G_Rxy1_Diagonal',
        'B_Rxy1_Horizontal', 'B_Rxy1_Vertical', 'B_Rxy1_Diagonal',
        'R_Rxy2_Horizontal', 'R_Rxy2_Vertical', 'R_Rxy2_Diagonal',
        'G_Rxy2_Horizontal', 'G_Rxy2_Vertical', 'G_Rxy2_Diagonal',
        'B_Rxy2_Horizontal', 'B_Rxy2_Vertical', 'B_Rxy2_Diagonal',
        'R_npcr1', 'G_npcr1', 'B_npcr1',
        'R_uaci1', 'G_uaci1', 'B_uaci1',
        'R_npcr2', 'G_npcr2', 'B_npcr2',
        'R_uaci2', 'G_uaci2', 'B_uaci2'
    ])

    df.to_excel("image_analysis_results_30-256.xlsx", index=False)


if __name__ == '__main__':
    main()
