#p77
import torch
import torch.nn as nn
import numpy as np
import einops.layers.torch as elt
#三个卷积层作为特征提取层
#全连接层作为特征分类层
class MnistNetword(nn.Module):
    def __init__(self):
        super().__init__()
        #前置的特征提取模块
        self.convs_stack = nn.Sequential(
            nn.Conv2d(1,12,kernel_size=7),
            nn.ReLU(),
            nn.Conv2d(12,24,kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(24,6,kernel_size=3)
        )
        #最终分类器层(logits_layer输出层)
        self.logits_layer = nn.Linear(in_features=1536,out_features=10)
    def forward(self, inputs):
        image = inputs
        x = self.convs_stack(image)
        #elt.Rearrange对输入数据的维度进行调整，也可以用torch.nn.Flatten
        x = elt.Rearrange("b c h w -> b (c h w)")(x)
        logits = self.logits_layer(x)
        return logits
if __name__ == '__main__':
    model = MnistNetword()
    torch.save(model,"model-27.pth")