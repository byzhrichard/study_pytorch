#p31 基于pytorch的MNIST手写体识别模型
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    #指定GPU编号
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
batch_size = 320    #每次训练的批次数
epochs = 1024       #训练次数
device = "cpu"
# device = "cuda"

#多层感知机
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_rule_stack = nn.Sequential(
            nn.Linear(28*28,312),
            nn.ReLU(),
            nn.Linear(312,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
    def forward(self, input):
        x = self.flatten(input) #展平为行向量
        logits = self.linear_rule_stack(x)
        return logits
model = NeuralNetwork()
model = model.to(device)        #将模型传入硬件等待计算
# model = torch.compile(model)    #加速计算速度
loss_fu = torch.nn.CrossEntropyLoss()                       #设定损失函数
optimizer = torch.optim.Adam(model.parameters(),lr=2e-5)    #设定优化函数

#载入数据
x_train = np.load("../../dataset/mnist/x_train.npy")
y_train_label = np.load("../../dataset/mnist/y_train_label.npy")
train_num = len(x_train)//batch_size

#开始计算
for epochs in range(20):
    train_loss = 0
    for i in range(train_num):
        start = i * batch_size
        end = (i + 1) * batch_size
        train_batch = torch.tensor(x_train[start:end]).to(device)
        label_batch = torch.tensor(y_train_label[start:end]).to(device)
        pred = model(train_batch)
        loss = loss_fu(pred, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()   #记录每个批次的损失值

    #计算并打印损失值
    train_loss /= train_num
    accuracy = (pred.argmax(1) == label_batch).type(torch.float32).sum().item() / batch_size
    print("train_loss:", round(train_loss, 2), "accuracy:", round(accuracy, 2))