#p28
import numpy as np
import torch
x_train = np.load("../../dataset/mnist/x_train.npy")
y_train_label = np.load("../../dataset/mnist/y_train_label.npy")
x = torch.tensor(y_train_label[:5],dtype=torch.int64)
y = torch.nn.functional.one_hot(x, 10)
print(y)
'''
tensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],         5
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],         0
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],         4
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],         1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])        9
'''

