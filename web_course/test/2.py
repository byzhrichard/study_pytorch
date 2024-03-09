import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from model import ResNet
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 判断是cpu还是gpu运行
print("using {} device.".format(device))

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 变换成（224，224）满足ResNet的输入
    transforms.ToTensor(),  # 变成Tensor，改变通道顺序等等
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 训练集 5W 张图片
trainset = torchvision.datasets.CIFAR10(root='../cifar', train=True, download=True, transform=data_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=True)

# 测试集 1W 张图片
testset = torchvision.datasets.CIFAR10(root='../cifar', train=False, download=True, transform=data_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=36, shuffle=False)

# CIFAR10 十个分类类别的label
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = ResNet()  # 载入网络
net.to(device)
loss_function = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 定义优化器

best_acc = 0.0
save_path = './resNet34.pth'  # 网络权重文件保存路径

for epoch in range(5):
    # train
    net.train()
    running_loss = 0.0

    for step, data in enumerate(trainloader, start=0):
        images, labels = data
        optimizer.zero_grad()  # 梯度清零
        out = net(images.to(device))  # 前向传播
        loss = loss_function(out, labels.to(device))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降

        running_loss += loss.item()  # 损失值

    # test
    net.eval()
    acc = 0.0
    total = 0
    with torch.no_grad():

        for test_data in testset:
            test_images, test_labels = test_data  # 取出测试集的image和label
            print(test_images.shape)
            outputs = net(test_images.to(device))  # 前向传播
            predict_y = torch.max(outputs, dim=1)[1]  # 取出最大的预测值
            acc += (predict_y == test_labels.to(device)).sum().item()  # 正确 +1
            total += test_labels.size(0)

    accurate = acc / total  # 计算整个test上面的正确率
    print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
          (epoch + 1, running_loss / step, accurate))

    if accurate > best_acc:
        best_acc = accurate
        torch.save(net.state_dict(), save_path)

print('Finished Training')

