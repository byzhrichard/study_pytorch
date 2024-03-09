#p84 下面是自定义的深度可分离膨胀卷积的定义
import torch
import torch.nn as nn
import numpy as np
import einops.layers.torch as elt

depth_conv = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, groups=6, dilation=2)
point_conv = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=1)
depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)

class MnistNetword(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs_stack = nn.Sequential(
            nn.Conv2d(1,12,kernel_size=7),
            nn.ReLU(),
            depthwise_separable_conv,   #使用自定义卷积代替了原生卷积层
            nn.ReLU(),
            nn.Conv2d(24,6,kernel_size=3)
        )
        self.logits_layer = nn.Linear(in_features=1536, out_features=10)
    def forward(self, inputs):
        image = inputs
        x = self.convs_stack(image)
        #elt.Rearrange对输入数据的维度进行调整，也可以用torch.nn.Flatten
        x = elt.Rearrange("b c h w -> b (c h w)")(x)
        logits = self.logits_layer(x)
        return logits