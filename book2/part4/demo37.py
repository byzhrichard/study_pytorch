import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
#安装tensorboard和tensorboardX
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_rule_stack = nn.Sequential(
            nn.Linear(28*28,312),
            nn.ReLU(),
            nn.Linear(312,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
    def forward(self, input):
        x = self.flatten(input)
        logits = self.linear_rule_stack(x)
        return logits
if __name__ == '__main__':
    model = NeuralNetwork()

    input_data = (torch.rand(5, 784))
    writer = SummaryWriter()
    with writer:
        writer.add_graph(model,input_data)

