#自定义nn模块
#pytorch优化模块optim的调用
import torch.nn
import pytorch
class TwoLayerNet(torch.nn.Module):     #继承torch.nn.Module
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):   #重写forward方法
        """
        在前向传播的函数中,接收一个输入张量, 必须返回一个输出张量
        可以使用构造函数中定义的模块以及张量上的任意(可微分)操作
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

myNet = TwoLayerNet(D_in, H, D_out)

loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(myNet.parameters(), lr=1e-4)#SGD优化算法

for t in range(500):
    y_pred = myNet(x)

    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
