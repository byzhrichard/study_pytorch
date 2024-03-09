#pytorch建立神经网络
import torch

dtype = torch.float
device = torch.device("cpu")
#如果要在GPU上运行,就使用以下代码
#device = torch.device("cuda: 0")

#N:批量大小  D_in:输入维度  D_H:隐藏维度  D_out:输出维度
N, D_in, D_H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)    #x: (64, 1000)
y = torch.randn(N, D_out, device=device, dtype=dtype)   #y: (64, 10)

#初始化权重(就是起到全连接层Linear的作用)
w1 = torch.randn(D_in, D_H, device=device, dtype=dtype)     #w1: (1000, 100)
w2 = torch.randn(D_H, D_out, device=device, dtype=dtype)    #w2: (100, 10)

learning_rate = 1e-6

for t in range(500):
    #隐藏层:h
    #pytorch的矩阵乘法: .mm()方法
    h = x.mm(w1)                            #h(64, 100)
    #clamp()方法检查每个元素是否超出指定范围。如果超出，将其替换为相应范围的最大/最小值。
    h_relu= h.clamp(min=0)#对h进行ReLU
    y_pred = h_relu.mm(w2)

    #pow(2)表示平方 sum()表示加和且只有一个维度,item()表示获取其中的元素
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 0:
        print(t, loss)

    #这也是梯度的算法, 不过我的评价是不如Adam算法
    grad_y_pred = 2.0 * (y_pred - y)        #梯度借用这个表示,越接近->梯度越小
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())    #t.()使得其可以反向放大(10->100)
    grad_h = grad_h_relu.clone()            #grad_h(64, 100)
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)

    #更新参数
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2


#x: (64, 1000)
#y: (64, 10)
#w1: (1000, 100)
#w2: (100, 10)
#h(64, 100)     h=relu(x*w1)
'''
y_pred = relu(x*w1) * w2                 (64, 10)

grad_y_pred = y_pred - y

grad_w2 = relu(x*w1) * grad_y_pred
grad_w1 = relu(w2*grad_y_pred) * x
'''