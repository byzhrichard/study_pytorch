# p90
import torch
import numpy as np

class ToTensor:
    def __call__(self, inputs, targets):
        return torch.tensor(inputs), torch.tensor(targets)
class MNIST_Dateset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.x_train = np.load("../../dataset/mnist/x_train.npy")
        self.y_train_label = np.load("../../dataset/mnist/y_train_label.npy")
        self.transform = transform      #显式地提供transform类
    def __getitem__(self, item):
        image = self.x_train[item]
        label = self.y_train_label[item]

        if self.transform:  #如果self.transform对象存在，则为真
            image, label = self.transform(image,label)

        return image,label

    def __len__(self):
        return len(self.y_train_label)
if __name__ == '__main__':
    mnist_dataset = MNIST_Dateset()
    image,label = (mnist_dataset[1024])
    print(type(image),type(label))  #<class 'numpy.ndarray'> <class 'numpy.uint8'>

    print("------------------")

    mnist_dataset = MNIST_Dateset(transform=ToTensor())
    image, label = (mnist_dataset[1024])
    print(type(image), type(label)) #<class 'torch.Tensor'> <class 'torch.Tensor'>
