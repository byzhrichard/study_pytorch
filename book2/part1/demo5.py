import numpy as np

x_train = np.load("../dataset/mnist/x_train.npy")
y_train_label = np.load("../dataset/mnist/y_train_label.npy")
print(y_train_label[:10])   #[5 0 4 1 9 2 1 3 1 4]

