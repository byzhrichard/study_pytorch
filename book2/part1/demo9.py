#p31 Adam优化器
import torch.optim
import demo7

model = demo7.NeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

