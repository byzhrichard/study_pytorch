#p24
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    #指定GPU
import torch
import numpy as np
import demo3
import matplotlib.pyplot as plt
from tqdm import tqdm
batch_size = 320        #设定每次训练的批次数
epochs = 1024           #设定训练次数
#device = "cpu"         #pytorch
device = "cuda"
model = demo3.Unet()
model = model.to(device)
model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(),lr=2e-5)

x_train = np.load("../dataset/mnist/x_train.npy")
y_train_label = np.load("../dataset/mnist/y_train_label.npy")
x_train_batch = []
for i in range(len(y_train_label)):
    if y_train_label[i] < 2:
        x_train_batch.append(x_train[i])
x_train = np.reshape(x_train_batch,[-1,1,28,28])
x_train /= 512.
train_length = len(x_train) * 20
for epoch in range(epochs):
    train_num = train_length//batch_size
    train_loss = 0
    for i in tqdm(range(train_num)):
        x_imgs_batch = []
        y_step_batch = []
        y_batch = []
        for b in range(batch_size):
            img = x_train[np.random.randint(x_train.shape[0])]
            x = img
            y = img
            x_imgs_batch.append(x)
            y_batch.append(y)
        x_imgs_batch = torch.tensor(x_imgs_batch).float().to(device)
        y_batch = torch.tensor(y_batch).float().to(device)
        pred = model(x_imgs_batch)
        loss = torch.nn.MSELoss(reduction=True)(pred,y_batch)/batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= train_loss
    print("train_loss:",train_loss)
    image = x_train[np.random.randint(x_train.shape[0])]
    image = np.reshape(image,[1,1,28,28])
    image = torch.tensor(image).float().to(device)
    image = model(image)
    image = torch.reshape(image,shape=[28,28])
    image = image.detach().cpu().numpy()

    plt.imshow(image)
    plt.savefig(f"./img/img_{epoch}.jpg")