#p73 全局池化层
import torch
image = torch.randn(size=(5,3,28,28))
image_pooled = torch.nn.AdaptiveAvgPool2d(1)(image)
print(image_pooled.shape)   #torch.Size([5, 3, 1, 1])