import torch.nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):  # 搭建 残差结构
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            # stride 不是1，因为有的残差块第一层 的stride = 2；对应残差块的虚线实线
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            # 第二个卷积核的stride = 1
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),  # 输入和输出的特征图size一样
            nn.BatchNorm2d(outchannel),
        )
        self.right = shortcut  # 捷径块

    def forward(self, x):
        out = self.left(x)
        # 实线，特征图的输入和输出size一样；虚线，shortcut部分要经过1*1卷积核改变特征图size
        residual = x if self.right is None else self.right(x)
        out += residual  # 加完之后在 ReLU
        out = F.relu(out)
        return out


class ResNet(nn.Module):  # 搭建ResNet 网络
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            # 输入：batch * 3 * 224 * 224
            # 输出特征图size = (224 - 7 + 2*3) / 2 + 1 = 112 +0.5 = 112 向下取整
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 输出 [batch,64,112,112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 输出特征图size = (112 - 3 + 2*1) / 2 + 1 = 56 +0.5 = 56 向下取整
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 输出 [batch,64,56,56]
        )

        self.layer1 = self._make_layer(64, 3, stride=1)  # conv2_x 有三个残差块
        self.layer2 = self._make_layer(128, 4, stride=2)  # conv3_x 有四个残差块
        self.layer3 = self._make_layer(256, 6, stride=2)  # conv4_x 有六个残差块
        self.layer4 = self._make_layer(512, 3, stride=2)  # conv5_x 有三个残差块

        self.fc = nn.Linear(512, num_classes)  # 分类层

    def _make_layer(self, channel, block_num, stride=1):  # 构建layer，每个层包含3 4 6 3 个残差块block_num
        shortcut = None
        layers = []  # 网络层

        if stride != 1:  # conv3/4/5_x 的第一层 stride 都是2，并且shortcut 需要特殊操作
            # 定义shortcut 捷径，都是1*1 的kernel ，需要保证和left最后相加的shape一样
            shortcut = nn.Sequential(
                nn.Conv2d(int(channel / 2), channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel)
            )
            layers.append(ResidualBlock(int(channel / 2), channel, stride, shortcut))
        else:
            layers.append(ResidualBlock(channel, channel, stride, shortcut))

        for i in range(1, block_num):  # 残差块后面几层的卷积是一样的
            layers.append(ResidualBlock(channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)  # batch*64*56*56
        x = self.layer1(x)  # batch*64*56*56
        x = self.layer2(x)  # batch*128*28*28
        x = self.layer3(x)  # batch*256*14*14
        x = self.layer4(x)  # batch*512*7*7
        x = F.avg_pool2d(x, 7)  # batch*512*1*1
        x = x.view(x.size(0), -1)  # x.size(0)是batch ，保持batch，其余的压成1维
        x = self.fc(x)  # batch*10(分类的个数)
        return x


model = ResNet()
# import torch
# input = torch.randn((10,3,224,224))
# model(input)


# 计算网络参数个数
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

