import torch
from torch import nn
from block.InnBlock import Noise_INN_block


class Noise_INN(nn.Module):
    def __init__(self):
        super(Noise_INN, self).__init__()

        self.inv1 = Noise_INN_block()
        self.inv2 = Noise_INN_block()
        self.inv3 = Noise_INN_block()
        self.inv4 = Noise_INN_block()
        self.inv5 = Noise_INN_block()
        self.inv6 = Noise_INN_block()
        self.inv7 = Noise_INN_block()
        self.inv8 = Noise_INN_block()
        self.inv9 = Noise_INN_block()
        # self.inv10 = Noise_INN_block()
        # self.inv11 = Noise_INN_block()
        # self.inv12 = Noise_INN_block()
        # self.inv13 = Noise_INN_block()
        # self.inv14 = Noise_INN_block()
        # self.inv15 = Noise_INN_block()

    def forward(self, x, rev=False):

        if not rev:
            input1, input2 = x[0], x[1]
            out = self.inv1(x)
            out[0] = out[0] + input1
            out = self.inv2(out)
            out[0] = out[0] - input1
            out = self.inv3(out)
            out[0] = out[0] + input1
            out = self.inv4(out)
            out[0] = out[0] - input1
            out = self.inv5(out)
            out[0] = out[0] - input1
            out = self.inv6(out)
            out[0] = out[0] - input1
            out = self.inv7(out)
            out[0] = out[0] - input1
            out = self.inv8(out)
            out[0] = out[0] - input1
            out = self.inv9(out)

        else:
            input1, input2 = x[0], x[1]
            # out = self.inv15(x, rev=True)
            # out[0] = out[0] + input1
            # out[1] = out[1] + input1
            # out = self.inv14(out, rev=True)
            # out[0] = out[0] - input1
            # out[1] = out[1] - input1
            # out = self.inv13(out, rev=True)
            # out[0] = out[0] + input1
            # out[1] = out[1] + input1
            # out = self.inv12(out, rev=True)
            # out[0] = out[0] - input1
            # out[1] = out[1] - input1
            # out = self.inv11(out, rev=True)
            # out[0] = out[0] + input1
            # out[1] = out[1] + input1
            # out = self.inv10(x, rev=True)
            # out[0] = out[0] - input1
            # out[1] = out[1] - input1
            out = self.inv9(x, rev=True)
            out[1] = out[1] + input1
            out = self.inv8(out, rev=True)
            out[1] = out[1] - input1
            out = self.inv7(out, rev=True)
            out[1] = out[1] + input1
            out = self.inv6(out, rev=True)
            out[1] = out[1] - input1
            out = self.inv5(out, rev=True)
            out[1] = out[1] + input1
            out = self.inv4(out, rev=True)
            out[1] = out[1] - input1
            out = self.inv3(out, rev=True)
            out[1] = out[1] + input1
            out = self.inv2(out, rev=True)
            out[1] = out[1] - input1
            out = self.inv1(out, rev=True)
        return out
