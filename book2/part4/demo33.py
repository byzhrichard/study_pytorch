# p89   举例
import torch
import numpy as np
class MNIST_Dateset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.x_train = np.load("../../dataset/mnist/x_train.npy")
        self.y_train_label = np.load("../../dataset/mnist/y_train_label.npy")
    def __getitem__(self, item):
        image = self.x_train[item]
        label = self.y_train_label[item]
        return image,label

    def __len__(self):
        return len(self.y_train_label)

