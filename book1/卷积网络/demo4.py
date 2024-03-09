#pytorch网络权重共享
import torch.nn
import random
import pytorch

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.inputs_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.inputs_linear(x).clamp(min=0)
        #对h_relu多次执行线性变换
        #动态层数的设计可以使网络更适应不同的任务和数据集，从而提高泛化能力
        for _ in range(random.randint(0,3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)    #多次使用相同的权重来计算
        y_pred = self.output_linear(h_relu)
        return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = DynamicNet(D_in, H, D_out)


criterion = torch.nn.MSELoss(reduction='sum')   #criterion:模型评估损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)  #optim优化器(SGD方法)
for t in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    if t % 100 == 0:
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

