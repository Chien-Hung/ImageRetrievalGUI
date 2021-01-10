import torch
import os
import torchvision.models as models
import torch.nn as nn
import numpy as np
import pdb


class ResNet50(nn.Module):
    def __init__(self, embed_dim=128):
        super(ResNet50, self).__init__()
        
        # print('Utilizing pretrained weights!')
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


if __name__ == '__main__':
    m = ResNet50(embed_dim=128)

    ckpt_path = '/media/men2/disk2/git_justplay/zzz/Deep-Metric-Learning-Baselines/Training_Results/cars196/CARS196_RESNET50_2020-12-2-10-42-37/checkpoint.pth.tar'

    if os.path.exists(ckpt_path):
        print('Loading checkpoints from {} ...'.format(ckpt_path))
        state_dict = torch.load(ckpt_path)['state_dict']
        m.load_state_dict(state_dict)

    print('done')
