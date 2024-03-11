'''ResNet-18 Image classfication for cifar1-10 with PyTorch

Author 'Sun-qian'.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),  #需要使用stride减少参数量
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),  #不变形
            nn.BatchNorm2d(outchannel)
        )
        #两个作用：使channel相同 and stride!=1时修为同形
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:  #当stride=1时不做，虽然做也行，但是会增加无用参数，使得效果变差
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, stride):
        strides = [stride] + [1]   #strides=[1,1]or[2,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        # print(len(layers))
        return nn.Sequential(*layers)
        #在 Python 中，星号 * 用作解包运算符（unpacking operator）。
        #当你在一个函数调用中使用 *layers，这意味着你将 layers 这个列表中的所有元素解包并作为单独的参数传递给函数。

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        print(out.shape)
        return out
def ResNet188():
    return ResNet(ResidualBlock)
def main():
    x = torch.randn(2,3,32,32)
    model = ResNet188()
    out = model(x)
if __name__ == '__main__':
    main()