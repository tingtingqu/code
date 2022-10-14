# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from torchvision.models import vgg19


class FCN8s(nn.Module):
    def __init__(self, num_classes=2):
        # ��ʼ������
        super().__init__()
        self.num_classes = num_classes
        model_vgg19 = vgg19(pretrained=True)
        self.base_model = model_vgg19.features
        # ���弸����Ҫ�Ĳ����
        self.relu = nn.ReLU(inplace=True) # ����ReLU�����
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, 
                                         padding=1, dilation=1, output_padding=1) # ʹ��ת�þ��������ӳ�������ά
        self.bn1 = nn.BatchNorm2d(512) # ���׼�����ô�������ݺ���ķֲ�������ѵ���Ĺ���
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        # ����layers�Ĳ���
        self.layers = {"4": "maxpool_1", "9": "maxpool_2", "18": "maxpool_3", "27": "maxpool_4", "36": "maxpool_5"}
        
    def forward(self, x):
        # ���ô��������������ݽ���ǰ�򴫵�
        output = {}
        
        for name, layer in self.base_model._modules.items():
            # �ӵ�һ�㿪ʼ��ȡͼ�������
            x = layer(x)
            if name in self.layers:
                output[self.layers[name]] = x # �����layers����ָ�����������򱣴浽output��
                
        x5 = output["maxpool_5"] # size = (N, 512, x.H/32, x.W/32)
        x4 = output["maxpool_4"] # size = (N, 512, x.H/16, x.W/16)
        x3 = output["maxpool_3"] # size = (N, 256, x.H/8, x.W/8)
        # size = (N, 512, x.H/32, x.W/32)
        score = self.relu(self.deconv1(x5))
        # ��ӦԪ����ӣ�size = (N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)
        # size = (N, 256, x.H/8, x.W/8)
        score = self.relu(self.deconv2(score))
        # ��ӦԪ����ӣ�size = (N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)
        # size = (N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score)))
        # size = (N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        # size = (N, 32, x.H, x.W)
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)
        return score ## size = (N, n_class, x.H, x.W)
