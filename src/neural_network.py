
import torch
import torch.nn as nn
import config as config


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
        file_name = f"model_{iteration}_{simuations}.pt"
        torch.save(self.state_dict(), f"{config.MODEL_PATH}/{file_name}")

    @staticmethod
    def load_model(file_name: str) -> "NeuralNet":
        """Load the model from a file"""

        model = NeuralNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)

        loaded_model: "NeuralNet"= torch.load(f"{config.MODEL_PATH}/{file_name}")

        model.load_state_dict(loaded_model.state_dict())

        model = model.eval()

        return model
    
if __name__ == "__main__":
    model = NeuralNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)
    model.save_model(1, 1)
    model.load_model("model_1_1.pt")
