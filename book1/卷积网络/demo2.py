#pytorch的Sequential模块调用     -代替了矩阵相乘,ReLu
#pytorch优化模块optim的调用      -代替了梯度清零,参数更新,代替了w1,w2
import torch

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
#nn包 定义模型和损失函数
my_model = torch.nn.Sequential(
    #包括两个线性层（输入层到隐藏层，隐藏层到输出层）和一个ReLU激活函数
    #H是hidden的中间维度, D_in是dimension_in, D_out是dimension_out
    #model(x)做的就是让x进行一系列运算, 如与Linear点乘
    torch.nn.Linear(D_in, H),   #传入x后, 执行点乘                   #将D_in维映射到H维
    torch.nn.ReLU(),            #然后对结果ReLU                      #ReLU
    torch.nn.Linear(H, D_out),  #然后对结果再点乘, 最终的结果输出      #将H维映射到到D_out维
)

learning_rate = 1e-4

#使用optim包定义优化器(optimizer)更新模型的权重
#Adam优化算法----自适应矩估计的优化算法
#Adam构造函数的第一个参数告诉优化器应该更新哪些张量
#model.parameters()是一个生成器函数，用于获取my_model中所有可学习参数
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

#均方误差损失（MSELoss）
#reduction='sum'表示在计算损失时如何对各个通道求和
#均方误差(相减后平方),然后做sum处理
loss_fn = torch.nn.MSELoss(reduction='sum')

for t in range(500):
    #前向传播, 通过向模型输入x计算预测的y
    y_pred = my_model(x)
    #算loss并打印
    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print(t, loss.item())
    #反向传播之前, 使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)
    #在多次迭代过程中，梯度可能会累积起来。
    #如果不清零梯度，优化器将继续根据上一次迭代的梯度来更新权重
    optimizer.zero_grad()   #梯度清零

    #反向传播, 根据模型的参数计算loss的梯度
    #为优化器提供梯度信息。在Adam优化器中，优化器根据梯度信息来更新模型权重
    loss.backward()     #反向传播 计算梯度

    #optimizer.step()函数根据计算出的梯度更新模型参数。
    optimizer.step()    #根据梯度 更新参数



