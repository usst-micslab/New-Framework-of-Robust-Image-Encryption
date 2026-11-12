import torch
import torch.nn as nn
from utils.jpeg import *
from model_inn.Inn import Noise_INN
from block.Haar import HaarDownsampling


class INL(nn.Module):
    def __init__(self):
        super(INL, self).__init__()
        self.model = Noise_INN()
        self.haar = HaarDownsampling(3)

    def forward(self, x, rev=False):

        if not rev:
            ran_img, ori_img = x[0], x[1]
            ran_img_dwt = self.haar(ran_img)
            ori_img_dwt = self.haar(ori_img)
            input_data = [ran_img_dwt, ori_img_dwt]
            out_dwt = self.model(input_data)
            output1_dwt, output2_dwt = out_dwt[0], out_dwt[1] + ran_img_dwt
            output1 = self.haar(output1_dwt, rev=True)
            output2 = self.haar(output2_dwt, rev=True)
            output_data = [output1, output2, ran_img_dwt, output2_dwt]

        else:
            ran_img, input2 = x[0], x[1]
            input1_dwt = self.haar(ran_img)
            input2_dwt = self.haar(input2)
            input_data = [input1_dwt, input2_dwt - input1_dwt]
            out_dwt = self.model(input_data, rev=True)
            output1_dwt, output2_dwt = out_dwt[0], out_dwt[1]
            output1 = self.haar(output1_dwt, rev=True)
            output2 = self.haar(output2_dwt, rev=True)
            output_data = [output1, output2]
        return output_data


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = 0.01 * torch.randn_like(param)
            if split[-2] == 'conv5':
                param.data.fill_(0.)
