import torch
from torch import nn

# Eksempelkode som kjører på FasionMNIST dataset:
# URL: https://github.com/Knolaisen/alpha-zero-prosjekt/blob/Per-Ivar/examples/fashionMNIST.py

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.layer_horizontal = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=[1,7]),
            nn.BatchNorm2d(6),
            nn.ReLU(),
    #        nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_vertical = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=7, kernel_size=[6,1]),
            nn.BatchNorm2d(7),
            nn.ReLU(),
    #        nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        
        self.policy = nn.Sequential(
          nn.Softmax(dim=7)
        )
        
        self.value = nn.Sequential(
          nn.Sigmoid(dim=1)
        )
        
        # self.fc1 = nn.Linear(in_features=1600, out_features=64)      
        # self.drop = nn.Dropout2d(0.25)

    def forward(self, x):
        out = self.layer_horizontal(x)
        out = self.layer_vertical(out)
        out = self.layer_3x3(out)
        
        value = self.value(out)
        policy = self.policy(out)
        
        
        # out = self.flatten(out)
        # out = self.fc1(out)
    #    out = self.drop(out)
        
        return out
      

model = NeuralNetwork().to(device)
print(model)
print(sum(param.numel() for param in model.parameters()))

