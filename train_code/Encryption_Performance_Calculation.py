import os
import cv2
import math
import torch
import warnings
import numpy as np
from utils.jpeg import *
from loss_calculation import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model_inn.encoder_decoder import *
from torch.utils.data import DataLoader
from utils.jpeg import JpegSS, JpegTest
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter

# ================================<<环境设备设置>>================================ #
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
warnings.filterwarnings("ignore")


# ================================<<环境设备设置>>================================ #
def NPCR(img1, img2):
    '''
    计算像素数变化率
    '''
    # opencv颜色通道顺序为BGR
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    w, h, _ = img1.shape

    # 图像通道拆分
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)
    # 返回数组的排序后的唯一元素和每个元素重复的次数
    ar, num = np.unique((R1 != R2), return_counts=True)
    if len(num) == 1:
        R_npcr = 0
    else:
        R_npcr = (num[0] if ar[0] == True else num[1]) / (w * h)

    ar, num = np.unique((G1 != G2), return_counts=True)
    if len(num) == 1:
        G_npcr = 0
    else:
        G_npcr = (num[0] if ar[0] == True else num[1]) / (w * h)

    ar, num = np.unique((B1 != B2), return_counts=True)
    if len(num) == 1:
        B_npcr = 0
    else:
        B_npcr = (num[0] if ar[0] == True else num[1]) / (w * h)

    return R_npcr, G_npcr, B_npcr


def UACI(img1, img2):
    '''
    两张图像之间的平均变化强度
    '''
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    w, h, _ = img1.shape
    # 图像通道拆分
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)
    # 元素为uint8类型取值范围：0到255
    # print(R1.dtype)

    # 强制转换元素类型，为了运算
    R1 = R1.astype(np.int16)
    R2 = R2.astype(np.int16)
    G1 = G1.astype(np.int16)
    G2 = G2.astype(np.int16)
    B1 = B1.astype(np.int16)
    B2 = B2.astype(np.int16)

    sumR = np.sum(abs(R1 - R2))
    sumG = np.sum(abs(G1 - G2))
    sumB = np.sum(abs(B1 - B2))
    R_uaci = sumR / 255 / (w * h)
    G_uaci = sumG / 255 / (w * h)
    B_uaci = sumB / 255 / (w * h)

    return R_uaci, G_uaci, B_uaci


def hist(img):
    img = cv2.imread(img)
    B, G, R = cv2.split(img)
    # 转成一维
    R = R.flatten(order='C')
    G = G.flatten(order='C')
    B = B.flatten(order='C')

    # 结果展示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    plt.subplot(232)
    # plt.imshow(img[:,:,(2,1,0)])
    plt.hist(img.flatten(order='C'), bins=range(257), color='gray')
    plt.title('原图像')
    # 子图2，通道R
    plt.subplot(234)
    # imshow()对图像进行处理，画出图像，show()进行图像显示
    plt.hist(R, bins=range(257), color='red')
    plt.title('通道R')
    # plt.show()
    # 不显示坐标轴
    # plt.axis('off')

    # 子图3，通道G
    plt.subplot(235)
    plt.hist(G, bins=range(257), color='green')
    plt.title('通道G')
    # plt.show()
    # plt.axis('off')

    # 子图4，通道B
    plt.subplot(236)
    plt.hist(B, bins=range(257), color='blue')
    plt.title('通道B')
    # plt.axis('off')
    # #设置子图默认的间距
    plt.tight_layout()
    plt.show()


def RGB_correlation(channel, N):
    '''
    分别计算图像通道相邻像素的水平、垂直和对角线的相关系数并返回
    '''
    # 计算channel通道
    h, w = channel.shape
    # 随机产生pixels个[0,w-1)范围内的整数序列
    row = np.random.randint(0, h - 1, N)
    col = np.random.randint(0, w - 1, N)
    # 绘制相邻像素相关性图,统计x,y坐标
    x = []
    h_y = []
    v_y = []
    d_y = []
    for i in range(N):
        # 选择当前一个像素
        x.append(channel[row[i]][col[i]])
        # 水平相邻像素是它的右侧也就是同行下一列的像素
        h_y.append(channel[row[i]][col[i] + 1])
        # 垂直相邻像素是它的下方也就是同列下一行的像素
        v_y.append(channel[row[i] + 1][col[i]])
        # 对角线相邻像素是它的右下即下一行下一列的那个像素
        d_y.append(channel[row[i] + 1][col[i] + 1])
    # 三个方向的合到一起
    x = x * 3
    y = h_y + v_y + d_y

    # 结果展示
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    # plt.scatter(x,y)
    # plt.show()

    # 计算E(x)，计算三个方向相关性时，x没有重新选择也可以更改
    ex = 0
    for i in range(N):
        ex += channel[row[i]][col[i]]
    ex = ex / N
    # 计算D(x)
    dx = 0
    for i in range(N):
        dx += (channel[row[i]][col[i]] - ex) ** 2
    dx /= N

    # 水平相邻像素h_y
    # 计算E(y)
    h_ey = 0
    for i in range(N):
        h_ey += channel[row[i]][col[i] + 1]
    h_ey /= N
    # 计算D(y)
    h_dy = 0
    for i in range(N):
        h_dy += (channel[row[i]][col[i] + 1] - h_ey) ** 2
    h_dy /= N
    # 计算协方差
    h_cov = 0
    for i in range(N):
        h_cov += (channel[row[i]][col[i]] - ex) * (channel[row[i]][col[i] + 1] - h_ey)
    h_cov /= N
    h_Rxy = h_cov / (np.sqrt(dx) * np.sqrt(h_dy))

    # 垂直相邻像素v_y
    # 计算E(y)
    v_ey = 0
    for i in range(N):
        v_ey += channel[row[i] + 1][col[i]]
    v_ey /= N
    # 计算D(y)
    v_dy = 0
    for i in range(N):
        v_dy += (channel[row[i] + 1][col[i]] - v_ey) ** 2
    v_dy /= N
    # 计算协方差
    v_cov = 0
    for i in range(N):
        v_cov += (channel[row[i]][col[i]] - ex) * (channel[row[i] + 1][col[i]] - v_ey)
    v_cov /= N
    v_Rxy = v_cov / (np.sqrt(dx) * np.sqrt(v_dy))

    # 对角线相邻像素d_y
    # 计算E(y)
    d_ey = 0
    for i in range(N):
        d_ey += channel[row[i] + 1][col[i] + 1]
    d_ey /= N
    # 计算D(y)
    d_dy = 0
    for i in range(N):
        d_dy += (channel[row[i] + 1][col[i] + 1] - d_ey) ** 2
    d_dy /= N
    # 计算协方差
    d_cov = 0
    for i in range(N):
        d_cov += (channel[row[i]][col[i]] - ex) * (channel[row[i] + 1][col[i] + 1] - d_ey)
    d_cov /= N
    d_Rxy = d_cov / (np.sqrt(dx) * np.sqrt(d_dy))

    return h_Rxy, v_Rxy, d_Rxy, x, y


def correlation(img, N=5000):
    '''
    分别计算图像img的各通道相邻像素的相关系数，默认随机选取3000对相邻像素
    '''
    img = cv2.imread(img)
    h, w, _ = img.shape
    B, G, R = cv2.split(img)
    R_Rxy = RGB_correlation(R, N)
    G_Rxy = RGB_correlation(G, N)
    B_Rxy = RGB_correlation(B, N)

    # 结果展示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    plt.subplot(221)
    plt.imshow(img[:, :, (2, 1, 0)])
    plt.title('原图像')
    # 子图2
    plt.subplot(222)
    plt.scatter(R_Rxy[3], R_Rxy[4], s=1, c='red')
    plt.title('通道R')

    # 子图3
    plt.subplot(223)
    plt.scatter(G_Rxy[3], G_Rxy[4], s=1, c='green')
    plt.title('通道G')
    # 子图4
    plt.subplot(224)
    plt.scatter(B_Rxy[3], B_Rxy[4], s=1, c='blue')
    plt.title('通道B')
    plt.show()

    return R_Rxy[0:3], G_Rxy[0:3], B_Rxy[0:3]


def entropy(img):
    '''
    计算图像的信息熵
    '''
    img = cv2.imread(img)
    w, h, _ = img.shape
    B, G, R = cv2.split(img)
    gray, num1 = np.unique(R, return_counts=True)
    gray, num2 = np.unique(G, return_counts=True)
    gray, num3 = np.unique(B, return_counts=True)
    R_entropy = 0
    G_entropy = 0
    B_entropy = 0

    for i in range(len(gray)):
        p1 = num1[i] / (w * h)
        p2 = num2[i] / (w * h)
        p3 = num3[i] / (w * h)
        R_entropy -= p1 * (math.log(p1, 2))
        G_entropy -= p2 * (math.log(p2, 2))
        B_entropy -= p3 * (math.log(p3, 2))
    return R_entropy, G_entropy, B_entropy


def GVD_components(channel0, channel1):
    # 计算图像分量的GVD,参数为加密前后的2个分量二维矩阵
    AN0 = 0
    AN1 = 0
    w, h = channel0.shape
    for i in range(1, w - 1):
        for j in range(1, h - 1):
            # 加密前
            GN0 = 0
            GN0 += (channel0[i][j] - channel0[i - 1][j]) ** 2
            GN0 += (channel0[i][j] - channel0[i + 1][j]) ** 2
            GN0 += (channel0[i][j] - channel0[i][j - 1]) ** 2
            GN0 += (channel0[i][j] - channel0[i][j + 1]) ** 2
            GN0 /= 4
            # 加密后
            GN1 = 0
            GN1 += (channel1[i][j] - channel1[i - 1][j]) ** 2
            GN1 += (channel1[i][j] - channel1[i + 1][j]) ** 2
            GN1 += (channel1[i][j] - channel1[i][j - 1]) ** 2
            GN1 += (channel1[i][j] - channel1[i][j + 1]) ** 2
            GN1 /= 4
            AN0 += GN0
            AN1 += GN1
    AN0 /= ((w - 2) * (h - 2))
    AN1 /= ((w - 2) * (h - 2))
    gvd = (AN1 - AN0) / (AN1 + AN0)
    return gvd


def GVD(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    w, h, _ = img1.shape
    B1, G1, R1 = cv2.split(img1)
    B2, G2, R2 = cv2.split(img2)
    R1 = R1.astype(np.int16)
    R2 = R2.astype(np.int16)
    G1 = G1.astype(np.int16)
    G2 = G2.astype(np.int16)
    B1 = B1.astype(np.int16)
    B2 = B2.astype(np.int16)
    R_gvd = GVD_components(R1, R2)
    G_gvd = GVD_components(G1, G2)
    B_gvd = GVD_components(B1, B2)
    return R_gvd, G_gvd, B_gvd


def main():
    original_img = './test_image/key1/encrypted_4.png'
    encrypted_img1 = './encrypted_image/key1/encrypted_4.png'
    encrypted_img2 = './encrypted_image/key2/encrypted_4.png'

    R_npcr, G_npcr, B_npcr = NPCR(encrypted_img1, encrypted_img2)
    print('*********NPCR*********')
    # 百分数表示，保留小数点后4位
    print('Red  :{:.4%}'.format(R_npcr))
    print('Green:{:.4%}'.format(G_npcr))
    print('Blue :{:.4%}'.format(B_npcr))

    R_uaci, G_uaci, B_uaci = UACI(encrypted_img1, encrypted_img2)
    print('*********UACI*********')
    # 百分数表示，保留小数点后4位
    print('Red  :{:.4%}'.format(R_uaci))
    print('Green:{:.4%}'.format(G_uaci))
    print('Blue :{:.4%}'.format(B_uaci))

    hist(original_img)
    hist(encrypted_img1)

    # 相关系数
    R_Rxy, G_Rxy, B_Rxy = correlation(original_img)
    # 输出结果保留四位有效数字
    print("******原始图像的各通道各方向的相关系数为*****")
    print('通道 \tHorizontal \tVertical \tDiagonal')
    print(' R    \t{:.4f}    {:.4f}    {:.4f}'.format(R_Rxy[0], R_Rxy[1], R_Rxy[2]))
    print(' G    \t{:.4f}    {:.4f}    {:.4f}'.format(G_Rxy[0], G_Rxy[1], G_Rxy[2]))
    print(' B    \t{:.4f}    {:.4f}    {:.4f}'.format(B_Rxy[0], B_Rxy[1], B_Rxy[2]))

    R_Rxy, G_Rxy, B_Rxy = correlation(encrypted_img1)
    # 输出结果保留四位有效数字
    print("******该图像的各通道各方向的相关系数为*****")
    print('通道 \tHorizontal \tVertical \tDiagonal')
    print(' R    \t{:.4f}    {:.4f}    {:.4f}'.format(R_Rxy[0], R_Rxy[1], R_Rxy[2]))
    print(' G    \t{:.4f}    {:.4f}    {:.4f}'.format(G_Rxy[0], G_Rxy[1], G_Rxy[2]))
    print(' B    \t{:.4f}    {:.4f}    {:.4f}'.format(B_Rxy[0], B_Rxy[1], B_Rxy[2]))

    # 图像的熵
    R_entropy, G_entropy, B_entropy = entropy(encrypted_img1)
    print('***********信息熵*********')
    print('通道R:{:.4}'.format(R_entropy))
    print('通道G:{:.4}'.format(G_entropy))
    print('通道B:{:.4}'.format(B_entropy))

    # 计算GVD
    R_gvd, G_gvd, B_gvd = GVD(original_img, encrypted_img1)
    print('***********GVD*********')
    print('通道R:{:.4}'.format(R_gvd))
    print('通道G:{:.4}'.format(G_gvd))
    print('通道B:{:.4}'.format(B_gvd))


if __name__ == '__main__':
    main()
