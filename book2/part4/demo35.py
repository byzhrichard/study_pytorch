# p92 错误示范
# 这样数据在输出时是逐个输出的，模型逐个数据计算损失函数时无法对其进行计算
import numpy as np
import torch

class ToTensor:
    def __call__(self, inputs, targets):
        inputs = np.reshape(inputs,[1,-1])
        targets = np.reshape(targets, [1,-1])
        return torch.tensor(inputs), torch.tensor(targets)
class MNIST_Dateset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.x_train = np.load("../../dataset/mnist/x_train.npy")
        self.y_train_label = np.load("../../dataset/mnist/y_train_label.npy")
        self.transform = transform
    def __getitem__(self, index):
        image = self.x_train[index]
        label = self.y_train_label[index]

        if self.transform:
            image, label = self.transform(image,label)

        return image,label

    def __len__(self):
        return len(self.y_train_label)



class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28,312),
            torch.nn.ReLU(),
            torch.nn.Linear(312,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,10)
        )
    def forward(self, input):
        x = self.flatten(input)
        logits = self.linear_relu_stack(x)

        return logits

device = "cuda"
mnist_dataset = MNIST_Dateset(transform=ToTensor())
# epochs = 1024
# batch_size = 320

model = NeuralNetwork()
model = model.to(device)
torch.save(model, './model-35.pth')
loss_fu = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
for epoch in range(20):
    train_loss = 0
    for sample in (mnist_dataset):  #通过这种方式逐一获取到每对image和label
        image = sample[0]
        label = sample[1]
        train_image = image.to(device)
        train_label = label.to(device)

        pred = model(train_image)
        loss = loss_fu(pred,train_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(mnist_dataset)
    print("epoch: ",epoch,"train_loss: ",round(train_loss,2))