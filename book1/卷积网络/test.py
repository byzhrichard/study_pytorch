import torch

import pytorch
a = torch.rand(5,10)
var = torch.nn.Linear(10,10)
b = var(a)
print(a)
print(b)
