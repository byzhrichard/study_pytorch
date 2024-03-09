import numpy as np
import math
def softmax(intMatrix):
    m, n = np.shape(intMatrix)
    outMatrix = np.mat(np.zeros((m,n)))
    soft_sum = 0
    for idx in range(0,n):
        outMatrix[0,idx] = math.exp(intMatrix[0,idx])
        soft_sum += outMatrix[0, idx]
    for idx in range(0,n):
        outMatrix[0,idx] = outMatrix[0,idx] / soft_sum
    return outMatrix
a = np.array([[1,2,1,2,1,1,3]])
print(softmax(a))
#[[0.05943317 0.16155612 0.05943317 0.16155612 0.05943317 0.05943317 0.43915506]]

