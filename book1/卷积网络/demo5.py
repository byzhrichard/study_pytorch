#全连接网络
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#定义网络结构
class Net(torch.nn.Module):
    def __init__(self, n_features):
        #nn.BatchNorm1d使数据均值为0,方差为1
        #nn.Sigmoid使数据归一化到0和1之间
        super(Net, self).__init__()     #继承__init__功能
        self.l1 = nn.Linear(n_features, 500)    #特征输入
        self.l2 = nn.ReLU()#增加非线性                       #激活函数
        self.l3 = nn.BatchNorm1d(500)                       #批归一化
        self.l4 = nn.Linear(500,250)
        self.l5 = nn.ReLU()
        self.l6 = nn.BatchNorm1d(250)
        self.l7 = nn.Linear(250,1)
        #self.l8 = nn.Sigmoid()
    def forward(self, inputs):
        #正向传播输入值, 神经网络分析出输出值
        out = torch.from_numpy(inputs).to(torch.float32)    #numpy -> tensor
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)
        # out = self.l8(out)
        return out

#超参数
torch.manual_seed(10)   #固定每次初始化模型的权重(从而使结果可重复)
training_step = 500     #迭代（训练）次数
batch_size = 512        #每个批次的大小
n_features = 32         #特征数目
M = 10000               #生成的数据数目

#生成数据
data = np.random.randn(M, n_features)
target = np.random.rand(M)

#特征归一化|数据预处理
min_max_scaler = MinMaxScaler()#创建MinMaxScaler对象
min_max_scaler.fit(data)#计算每个特征的最小值和最大值
data = min_max_scaler.transform(data)#将原始数据映射到0和1之间

#对训练集进行切割,然后进行训练    train:训练集   validation:验证集
x_train, x_val, y_train, y_val = train_test_split(data, target, test_size=0.2, shuffle=False)
#train_test_split函数将数据集划分为训练集和验证集
#训练集用于训练模型，而验证集用于在训练过程中评估模型的性能
#有20%的数据被划分为验证集val，剩余的80%被划分为训练集train
#shuffle=False表示不需要随机打乱数据
#x_train:(8000,32)
#x_val:(2000,32)
#y_train:(8000,)
#y_val:(2000,)

#定义模型
model = Net(n_features)
#定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)     #传入net的所有参数, 学习率
#定义目标损失函数
loss_func = torch.nn.MSELoss()      #均方差函数

#开始迭代(训练)
for _ in range(training_step):
    M_train = len(x_train)#8000
    with tqdm(np.arange(0, M_train, batch_size), desc='Training...') as tbar:
        for index in tbar:
            L = index
            R = min(M_train, index + batch_size)
            # 训练内容
            train_pre = model(x_train[L:R, :])  # 预测值
            train_loss = loss_func(train_pre,
                                   torch.from_numpy(y_train[L:R].reshape(R - L, 1)).to(torch.float32))
            val_pre = model(x_val)
            val_loss = loss_func(val_pre,
                                 torch.from_numpy(y_val.reshape(len(y_val), 1)).to(torch.float32))
            # 设置进度条的附录信息
            tbar.set_postfix(train_loss=float(train_loss.data),
                             val_loss=float(val_loss.data))
            tbar.update()  # 默认参数n=1, 每更新一次, 进度加n
            # 反向传播更新
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            train_loss.backward()  # 以训练集的误差进行反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到net的parameters上
