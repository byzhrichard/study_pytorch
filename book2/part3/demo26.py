#p77
import numpy as np
import einops.layers.torch as elt
#载入数据
x_train = np.load("../../dataset/mnist/x_train.npy")
y_train_label = np.load("../../dataset/mnist/y_train_label.npy")
print(x_train.shape)

x_train = np.expand_dims(x_train, axis=1)   #在第二维度进行扩充
print(x_train.shape)