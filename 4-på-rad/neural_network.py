import torch
from torch import nn
from env import ConnectFour


# Eksempelkode som kjører på FasionMNIST dataset:
# URL: https://github.com/Knolaisen/alpha-zero-prosjekt/blob/Per-Ivar/examples/fashionMNIST.py

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.layer_horizontal = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=[1, 7]).to(device),
            # nn.BatchNorm1d(6),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_vertical = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=7, kernel_size=[6, 1]).to(device),
            # nn.BatchNorm1d(7),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_4x4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, device=device),
            nn.BatchNorm1d(3, device=device),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten(0,-1)

        self.policy = nn.Sequential(
            nn.Linear(in_features=384, out_features=64, device=device),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32, device=device),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=7, device=device),
            nn.Softmax(dim=0),
            
        )

        self.value = nn.Sequential(
            nn.Linear(in_features=384, out_features=64, device=device),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32, device=device),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1, device=device),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = torch.from_numpy(x).to(device=device, dtype=torch.float)
        shape = 1, *game.shape
        out = out.view(shape)

        # out = self.layer_horizontal(x)
        # out = self.layer_vertical(out)
        out = self.layer_4x4(out)
        out = self.flatten(out)

        policy = self.policy(out).detach().cpu().numpy()
        value = self.value(out).detach().cpu().numpy()

        return policy, value


model = NeuralNetwork().to(device=device)
game = ConnectFour()

#state = game.get_board()

#policy, value = model.forward(state)
#print(forward)

#print(f'\n\nPolicy: {policy}\n Value:{value}')
