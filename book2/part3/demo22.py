#p71
import torch
image = torch.randn(size=(5,3,128,128)) #[batch,channel,height,width]
#卷积层示例
"""
输入维度：3
输出维度：10
卷积核大小：[3,3]
步长：2
补偿方式：维度不变
"""
conv2d = torch.nn.Conv2d(3,10,kernel_size=3,stride=1,padding=1)
image_new = conv2d(image)
print(image_new.shape)  #torch.Size([5, 10, 128, 128])

# result = (边-kernel+2*padding)//stride + 1

#padding=1表示图形边缘由一圈0补齐，成为torch.Size([5, 10, 126, 126])
#padding=0时，为torch.Size([5, 10, 126, 126])

#stride为2时，为torch.Size([5, 10, 64, 64])
#result = (128-3+2*1)//2 + 1 = 64