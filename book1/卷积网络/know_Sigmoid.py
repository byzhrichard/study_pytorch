import torch
import torch.nn as nn

m = nn.Sigmoid()

input = torch.randn(4)
output = m(input)

print("input: ", input)   # input:  tensor([-0.8462,  0.7929, -0.5680, -0.5883])
print("output: ", output) # output:  tensor([0.3002, 0.6885, 0.3617, 0.3570])
