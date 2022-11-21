import torch
from torch import nn
from env import ConnectFour


# Eksempelkode som kjører på FasionMNIST dataset:
# URL: https://github.com/Knolaisen/alpha-zero-prosjekt/blob/Per-Ivar/examples/fashionMNIST.py

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.layer_horizontal = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=[1, 7]).to(device),
            # nn.BatchNorm1d(6),
            nn.ReLU(),
            #        nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_vertical = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=7, kernel_size=[6, 1]).to(device),
            # nn.BatchNorm1d(7),
            nn.ReLU(),
            #        nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_4x4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4).to(device),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten(0,-1)

        self.policy = nn.Sequential(
            nn.Linear(in_features=384, out_features=64, device=device),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=7),
            nn.Softmax(dim=0),
            
        )

        self.value = nn.Sequential(
            nn.Linear(in_features=384, out_features=64, device=device),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

        # self.fc1 = nn.Linear(in_features=1600, out_features=64)
        # self.drop = nn.Dropout2d(0.25)

    def forward(self, x):
        # out = self.layer_horizontal(x)
        # out = self.layer_vertical(out)
        out = self.layer_4x4(x)
        out = self.flatten(out)

        policy = self.policy(out)
        value = self.value(out)

        # out = self.flatten(out)
        # out = self.fc1(out)
    #    out = self.drop(out)

        return policy, value


model = NeuralNetwork().to(device)
print(model)
# print(sum(param.numel() for param in model.parameters()))

game = ConnectFour()
input_tensor = torch.from_numpy(game.get_board()).to(device)
input_tensor = input_tensor.type(torch.FloatTensor)
# input_tensor = input_tensor.resize_(6, 7, 1)
input_tensor = input_tensor.view(1, 6, 7)

print(input_tensor.shape)

nn = NeuralNetwork()
forward = nn.forward(input_tensor)
print(forward)
