import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from ResNet import ResNet18
from net import ResNet188

def main():
    batch_size=100
    cifar_train = datasets.CIFAR10('../cifar', #访问当前目录的cifar文件夹
                             True,
                             transform=transforms.Compose([
                                 transforms.Resize((32,32)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485,0.456,0.406], #image-net的初始化，更好的性能
                                                      std=[0.229,0.224,0.225])
                             ]),
                             download=True) #如果没有，那么下载
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    # print("awa", len(cifar_train) * batch_size)
    cifar_test = datasets.CIFAR10('../cifar',
                                   False,
                                   transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                   ]),
                                   download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)
    # print("awa", len(cifar_test) * batch_size)

    # x, label =iter(cifar_train).__next__()
    # print("x.shape:",x.shape,"label.shape:",label.shape)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # model = ResNet18().to(device)
    model = ResNet188().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    # print(device)
    for epoch in range(20):
        f = open("train_out.txt", "a+")  # 追加，可读可写
        model.train()#训练模式
        loss_sum = []
        for batchidx, (x, label) in enumerate(cifar_train):
            if batchidx%50==0:print(batchidx)
            # [b,3,32,32]
            # [b]
            x, label = x.to(device), label.to(device)
            logits = model(x)
            # logits    [b,10]
            # label     [b]
            loss = criteon(logits,label)
            loss_sum.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #完成了一个epoch
        print(f"epoch:{epoch},loss:{np.mean(loss_sum)}")
        print(f"epoch:{epoch},loss:{np.mean(loss_sum)}", file=f)
        model.eval()#测试模式
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                #[b,10]
                logits = model(x)
                #[b]
                pred = logits.argmax(dim=1)
                #[b] vs [b]
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)
            acc = total_correct / total_num
            print(f"epoch:{epoch},acc:{acc}")
            print(f"epoch:{epoch},acc:{acc}", file=f)
            f.close()
if __name__ == '__main__':
    main()