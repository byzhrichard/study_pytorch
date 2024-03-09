#p53
import numpy as np
m = 20
x0 = np.ones((m ,1))
x1 = np.arange(1, m+1).reshape(m, 1)    #np.arange()包左不包右
x = np.hstack((x0, x1)) #并列操作，形状为[20,2]
y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)
alpha = 0.01
#这里theta是一个[2,1]大小的矩阵，用来与输入的x进行计算，以获得计算的预算值y_pred，而y_pred用于与y计算误差
def error_function(theta, x, y):
    h_pred = np.dot(x, theta)
    j_theta = (1./2*m) * np.dot(np.transpose(h_pred), h_pred)   #np.transpose()
    return j_theta
def gradient_function(theta, X, y):
    h_pred = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), h_pred)
def gradient_decent(X, u, alpha):
    theta = np.array([1, 1]).reshape(2, 1)  #[1,1]是theta的初始化参数，后面会修改
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-4):    #np.absolute()
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
        print(theta)
    return theta
theta = gradient_decent(x, y, alpha)
print('optimal:', theta)
print('error function:', error_function(theta, x, y)[0, 0])