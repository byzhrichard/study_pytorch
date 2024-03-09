#p78 基于卷积的MNIST手写体识别模型
import torch
import torch.nn as nn
import numpy as np
import einops.layers.torch as elt
#载入数据
x_train = np.load("../../dataset/mnist/x_train.npy")
y_train_label = np.load("../../dataset/mnist/y_train_label.npy")
x_train = np.expand_dims(x_train, axis=1)   #在第二维度进行扩充
print(x_train.shape)

class MnistNetword(nn.Module):
    def __init__(self):
        super().__init__()
        #前置的特征提取模块
        self.convs_stack = nn.Sequential(
            nn.Conv2d(1,12,kernel_size=7),
            nn.ReLU(),
            nn.Conv2d(12,24,kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(24,6,kernel_size=3)
        )
        #最终分类器层
        self.logits_layer = nn.Linear(in_features=1536,out_features=10)
    def forward(self, inputs):
        image = inputs
        x = self.convs_stack(image)
        #elt.Rearrange对输入数据的维度进行调整，也可以用torch.nn.Flatten
        x = elt.Rearrange("b c h w -> b (c h w)")(x)
        logits = self.logits_layer(x)
        return logits

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(str(device))
    # 注意记得将model发送到GPU计算
    model = MnistNetword().to(device)
    # model = torch.compile(model)    #加速计算速度
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    batch_size = 128
    for epoch in range(42):
        train_num = len(x_train) // 128
        train_loss = 0
        for i in range(train_num):
            start = i * batch_size
            end = (i + 1) * batch_size
            x_batch = torch.tensor(x_train[start:end]).to(device)
            y_batch = torch.tensor(y_train_label[start:end]).to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # 记录每个批次的损失值
        train_loss /= train_num
        accuracy = (pred.argmax(1) == y_batch).type(torch.float32).sum().item() / batch_size
        print("epoch: ", epoch, "train_loss: ", round(train_loss, 2), "accuracy: ", round(accuracy, 2))