import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        resnetAll = models.resnet50(pretrained=True)
        modules = list(resnetAll.children())[:-1]
        self.resnet = torch.nn.Sequential(*modules)


    def forward(self, inputs):

        x1 = self.resnet(inputs)
        x1 = x1.view(x1.size(0), -1)
        
        return x1
