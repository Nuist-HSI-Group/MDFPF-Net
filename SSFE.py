# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, num_bands):
        super().__init__()

        # 共享参数配置
        self.channel_config = {
            'large': [224, 128, 64, 32],  # 高光谱通道数>200时的配置
            'small': [128, 64, 32]  # 普通通道数的配置
        }

        # 动态选择配置
        config_key = 'large' if num_bands > 200 else 'small'
        channels = self.channel_config[config_key]

        # 空间特征分支构建器
        def build_spatial_branch(in_ch):
            layers = []
            for out_ch in channels:
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch, momentum=0.9),
                    nn.LeakyReLU(),
                ])
                in_ch = out_ch
            return nn.Sequential(*layers)

        # 光谱特征分支构建器
        def build_spectral_branch(in_ch):
            layers = []
            for out_ch in channels:
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, kernel_size=1),
                    nn.BatchNorm2d(out_ch, momentum=0.9),
                    nn.LeakyReLU(),
                ])
                in_ch = out_ch
            return nn.Sequential(*layers)

        # 初始化分支
        self.spatial_branch = build_spatial_branch(num_bands)
        self.spectral_branch = build_spectral_branch(num_bands)

    def forward(self, x):
        return self.spatial_branch(x), self.spectral_branch(x)

class AReLU(nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)

        return F.relu(input) * beta - F.relu(-input) * alpha