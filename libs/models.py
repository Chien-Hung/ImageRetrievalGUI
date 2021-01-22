import torch
import os
import torchvision.models as models
import torch.nn as nn
import numpy as np


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        embed_dim = 128
        self.model = models.resnet50(pretrained=None)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None
        
        self.model.last_linear = torch.nn.Linear(self.model.fc.in_features, embed_dim)
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        delattr(self.model, 'fc')

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        mod_x = self.model.last_linear(x)
        return torch.nn.functional.normalize(mod_x, dim=-1)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        
        self.model = models.resnet18(pretrained=True)
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return torch.nn.functional.normalize(x, dim=-1)
