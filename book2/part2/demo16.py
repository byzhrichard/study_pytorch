#p50
import numpy as np
x = 1
def chain(x, gama = 0.1):
    x = x - gama * 2 * x
    return x
for _ in range(4):
    x = chain(x)
    print(x)