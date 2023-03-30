import torch.nn as nn
import torch
from torchsummary import summary


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    

net = NeuralNet(input_size=10, hidden_size=20, output_size=5)
summary(net, input_size=(1, 10))
input_data = torch.randn(1, 10) # create a random input tensor
output = net(input_data) # pass the input through the network