#全连接网络
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#定义网络结构
class Net(torch.nn.Module):
    def __init__(self, n_features):
        #nn.BatchNorm1d使数据均值为0,方差为1
        #nn.Sigmoid使数据归一化到0和1之间
        super(Net, self).__init__()     #继承__init__功能
        self.l1 = nn.Linear(n_features, 500)    #特征输入
        self.l2 = nn.ReLU()                                 #激活函数
        self.l3 = nn.BatchNorm1d(500)                       #批归一化
        self.l4 = nn.Linear(500,250)
        self.l5 = nn.ReLU()
        self.l6 = nn.BatchNorm1d(250)
        self.l7 = nn.Linear(250,1)
        self.l8 = nn.Sigmoid()
    def forward(self, inputs):      #这同时也是Module中的正向传播功能
        #正向传播输入值, 神经网络分析出输出值
        out = torch.from_numpy(inputs).to(torch.float32)    #numpy -> tensor
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)
        out = self.l8(out)
        return out

#超参数
torch.manual_seed(10)   #固定每次初始化模型的权重
training_step = 500     #迭代（训练）次数
batch_size = 512        #每个批次的大小
n_features = 32         #特征数目
M = 10000               #生成的数据数目

#生成数据
data = np.random.randn(M, n_features)
target = np.random.rand(M)

min_max_scaler = MinMaxScaler()         #特征归一化
min_max_scaler.fit(data)
data = min_max_scaler.transform(data)

#对训练集进行切割,然后进行训练
x_train, x_val, y_train, y_val = train_test_split(data,target,test_size=0.2,shuffle=False)

#定义模型
model = Net(n_features)
print(model)