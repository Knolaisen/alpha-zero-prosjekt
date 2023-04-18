
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # 1 - input layer
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(hidden_size, output_size)  # 5 - output layer

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x):
        out = self.linear1(x)
        out = self.sigmoid1(out)
        out = self.linear2(out)
        out = self.sigmoid2(out)
        out = self.linear3(out)
        return out

    def save_model(self, iteration: int, simuations: int) -> None:
        """Save the model to a file"""
        pass

    @staticmethod
    def loadModel(file_name: str, model: "NeuralNet") -> "NeuralNet":
        """Load the model from a file"""
        pass