import torch
import torch.nn as nn

import torch.nn.functional as F
import cv2
import itertools
import numpy as np

torch.set_printoptions(threshold=300000)
from torch.autograd import Variable

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class FuzzyLayer(nn.Module):
    def __init__(self, fuzzynum, channel):
        super(FuzzyLayer, self).__init__()
        self.n = fuzzynum
        self.channel = channel
        self.conv1 = nn.Conv2d(self.channel, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(1, self.channel, 3, padding=1)
        self.mu = nn.Parameter(torch.randn((self.channel, self.n)))
        self.sigma = nn.Parameter(torch.randn((self.channel, self.n)))
        self.bn1 = nn.BatchNorm2d(1, affine=True)
        self.bn2 = nn.BatchNorm2d(self.channel, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        tmp = torch.tensor(np.zeros((x.size()[0], x.size()[1], x.size()[2], x.size()[3])), dtype=torch.float).to(device)
        for num, channel, w, h in itertools.product(range(x.size()[0]), range(x.size()[1]), range(x.size()[2]),
                                                    range(x.size()[3])):
            for f in range(self.n):
                tmp[num][channel][w][h] -= ((x[num][channel][w][h] - self.mu[channel][f]) / self.sigma[channel][f]) ** 2
        fNeural = self.bn2(self.conv2(self.bn1(torch.exp(tmp))))
        return fNeural


class Attention_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Attention_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                  nn.BatchNorm2d(out_ch), nn.Sigmoid())
        self.psi = nn.Sequential(nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1),
                                 nn.Sigmoid())
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.conv(g)
        x1 = self.conv(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Segnet_unet(nn.Module):
    def __init__(self, input_nbr, label_nbr):
        super(Segnet_unet, self).__init__()
        # 妯″潡1
        filters = [64, 128, 256, 512]
        self.conv11 = nn.Conv2d(input_nbr, 32, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(32, momentum=0.1)
        self.conv12 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv_u1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_u1 = nn.BatchNorm2d(64, momentum=0.1)

        # 妯″潡2
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv_u2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_u2 = nn.BatchNorm2d(128, momentum=0.1)
        # 妯″潡3
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=0.1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=0.1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=0.1)
        self.conv_u3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn_u3 = nn.BatchNorm2d(256, momentum=0.1)
        # 妯″潡4
        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv_u4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn_u4 = nn.BatchNorm2d(512, momentum=0.1)

        # 妯″潡5
        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=0.1)
        self.fbn5 = nn.BatchNorm2d(512, affine=True)
        # 鏈€涓棿鍔犱竴涓ā绯婂�?

        # 妯″潡6
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=0.1)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=0.1)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=0.1)

        # 妯″潡7
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=0.1)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=0.1)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=0.1)

        # 妯″潡8
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=0.1)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=0.1)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=0.1)

        # 妯″潡9

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=0.1)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=0.1)
        # 模块10
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=0.1)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # 注意力模�?
        self.fuzzy_5 = FuzzyLayer(fuzzynum=1, channel=512)
        self.at7 = Attention_block(in_ch=512, out_ch=256)
        self.at8 = Attention_block(in_ch=256, out_ch=128)

        self.at9 = Attention_block(in_ch=128, out_ch=64)
        self.at10 = Attention_block(in_ch=64, out_ch=32)


    def forward(self, input):
        # 妯″潡1
        out11 = self.conv11(input)
        out11 = self.bn11(out11)
        out11 = self.relu(out11)
        out12 = self.conv12(out11)
        out12 = self.bn12(out12)
        out12 = self.relu(out12)
        out1p, outid1 = F.max_pool2d(out12, kernel_size=2, stride=2, return_indices=True)
        # 妯″潡2
        out21 = self.conv21(out1p)
        out21 = self.bn21(out21)
        out21 = self.relu(out21)
        out22 = self.conv22(out21)
        out22 = self.bn22(out22)
        out22 = self.relu(out22)
        out2p, outid2 = F.max_pool2d(out22, kernel_size=2, stride=2, return_indices=True)
        # 妯″潡3
        out31 = self.conv31(out2p)
        out31 = self.bn31(out31)
        out31 = self.relu(out31)
        out32 = self.conv32(out31)
        out32 = self.bn32(out32)
        out32 = self.relu(out32)
        out33 = self.conv33(out32)
        out33 = self.bn33(out33)
        out33 = self.relu(out33)
        out3p, outid3 = F.max_pool2d(out33, kernel_size=2, stride=2, return_indices=True)
        # 妯″潡4
        out41 = self.conv41(out3p)
        out41 = self.bn41(out41)
        out41 = self.relu(out41)
        out42 = self.conv42(out41)
        out42 = self.bn42(out42)
        out42 = self.relu(out42)
        out43 = self.conv43(out42)
        out43 = self.bn43(out43)
        out43 = self.relu(out43)
        out4p, outid4 = F.max_pool2d(out43, kernel_size=2, stride=2, return_indices=True)

        # 妯″潡5
        out51 = self.conv51(out4p)
        out51 = self.bn51(out51)
        out51 = self.relu(out51)
        out52 = self.conv52(out51)
        out52 = self.bn52(out52)
        out52 = self.relu(out52)
        out53 = self.conv53(out52)
        out53 = self.bn53(out53)
        out53 = self.relu(out53)
        out5p, outid5 = F.max_pool2d(out53, kernel_size=2, stride=2, return_indices=True)
        h = out5p
        out5p = self.fbn5(self.fuzzy_5(out5p)) + h
        # 涓棿鍔犱竴涓ā绯婂�?

        # 妯″潡6
        out5d = F.max_unpool2d(out5p, outid5, kernel_size=2, stride=2)
        out53d = self.conv53d(out5d)
        out53d = self.bn53d(out53d)
        out53d = self.relu(out53d)
        out52d = self.conv52d(out53d)
        out52d = self.bn52d(out52d)
        out52d = self.relu(out52d)
        out51d = self.conv51d(out52d)
        out51d = self.bn51d(out51d)
        out51d = self.relu(out51d)

        # 7+娉ㄦ剰鍔?
        out4d = F.max_unpool2d(out51d, outid4, kernel_size=2, stride=2)
        out63d = self.conv43d(out4d)
        out63d = self.bn43d(out63d)
        out63d = self.relu(out63d)
        out62d = self.conv42d(out63d)
        out62d = self.bn42d(out62d)
        out62d = self.relu(out62d)

        out43 = self.at7(out62d, out43)
        merge7 = torch.cat([out62d, out43], dim=1)
        out_4d = self.conv_u4(merge7)
        out_4d = self.bn_u4(out_4d)
        out_4d = self.relu(out_4d)
        out61d = self.conv41d(out_4d)


        # 8
        out3d = F.max_unpool2d(out61d, outid3, kernel_size=2, stride=2)
        out73d = self.conv33d(out3d)
        out73d = self.bn33d(out73d)
        out73d = self.relu(out73d)
        out72d = self.conv32d(out73d)
        out72d = self.bn32d(out72d)
        out72d = self.relu(out72d)
        out33 = self.at8(out72d, out33)  # 256 128
        merge8 = torch.cat([out72d, out33], dim=1)
        out_3d = self.conv_u3(merge8)
        out_3d = self.bn_u3(out_3d)
        out_3d = self.relu(out_3d)
        out71d = self.conv31d(out_3d)


        # 模块9 outid2=128  out71d=128
        out2d = F.max_unpool2d(out71d, outid2, kernel_size=2, stride=2)
        out82d = self.conv22d(out2d) #128 128
        out82d = self.bn22d(out82d)
        out82d = self.relu(out82d)
        out22 = self.at9(out82d, out22) #128 128
        merge9 = torch.cat([out82d, out22], dim=1)
        out_2d = self.conv_u2(merge9)
        out_2d = self.bn_u2(out_2d)
        out_2d = self.relu(out_2d)
        out81d = self.conv21d(out_2d)

        #模块10
        out1d = F.max_unpool2d(out81d, outid1, kernel_size=2, stride=2)
        out92d = self.conv12d(out1d)
        out92d = self.bn12d(out92d)
        out92d = self.relu(out92d)
        out12 = self.at10(out92d, out12)
        merge10 = torch.cat([out92d, out12], dim=1)
        out_1d = self.conv_u1(merge10)
        out_1d = self.bn_u1(out_1d)
        out_1d = self.relu(out_1d)
        out = self.conv11d(out_1d)
        return out
