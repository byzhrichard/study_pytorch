# p100
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

class ToTensor:
    def __call__(self, inputs, targets):
        inputs = np.reshape(inputs,[28*28])
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
epochs = 320
batch_size = 320
train_loader = DataLoader(mnist_dataset, batch_size=batch_size)

model = NeuralNetwork()
model = model.to(device)
loss_fu = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)   #2乘以10的-6次方 故意调整了学习率
writer = SummaryWriter()
for epoch in range(epochs):
    train_loss = 0
    for image, label in (train_loader):
        train_image = image.to(device)
        train_label = label.to(device)

        pred = model(train_image)
        loss = loss_fu(pred,train_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss/batch_size
    print("epoch: ",epoch,"train_loss: ",round(train_loss,2))
    writer.add_scalars('evl', {'train_loss':train_loss}, epoch) #直接记录损失过程
writer.close()