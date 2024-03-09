#p29    多层感知机
import torch.nn as nn
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

    