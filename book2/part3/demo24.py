#p73
import torch
image = torch.randn(size=(5,3,28,28))   # [batch,channel,height,width]
pool = torch.nn.AvgPool2d(kernel_size=3,stride=2,padding=0)
image_pooled = pool(image)
print(image_pooled.shape)   #torch.Size([5, 3, 13, 13])