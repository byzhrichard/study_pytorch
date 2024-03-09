#p52
import numpy as np
def error_function(theta, x, y):
    h_pred = np.dot(x, theta)
    j_theta = (1./2*m) * np.dot(np.transpose(h_pred), h_pred)
    return j_theta
def gradient_function(theta, X, y):
    h_pred = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), h_pred)

#梯度下降的python实现
def gradient_descent(X, u, alpha):#可能造成欠下降or过下降
    theta = np.array([1, 1]).reshape(2, 1)  #[1,1]是theta的初始化参数，后面会修改
    gradient = gradient_function(theta, X, y)
    for i in range(17):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta
def gradient_decent(X, u, alpha):#可以设定阈值or停止条件
    theta = np.array([1, 1]).reshape(2, 1)  #[1,1]是theta的初始化参数，后面会修改
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-4):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
        print(theta)
    return theta