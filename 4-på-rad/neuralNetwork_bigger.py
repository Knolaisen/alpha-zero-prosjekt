import torch
from torch import nn

# Eksempelkode som kjører på FasionMNIST dataset:
# URL: https://github.com/Knolaisen/alpha-zero-prosjekt/blob/Per-Ivar/examples/fashionMNIST.py

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.layer_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2**8, kernel_size=3, device=device),
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
            nn.Tanh()
        )

    def forward(self, x, tensor=False, only_policy=False):
        out = torch.from_numpy(x).to(device=device, dtype=torch.float)
        #shape = 1, *game.shape
        out = out.view((1,6,7))

        # out = self.layer_horizontal(x)
        # out = self.layer_vertical(out)
        out = self.layer_4x4(out)
        out = self.flatten(out)

        policy = self.policy(out)
        if only_policy:
            return policy.detach().cpu().numpy()
        value = self.value(out)        

        if tensor: # returns tensor
            return policy, value

        # else returns numpy array
        return policy.detach().cpu().numpy(), value.detach().cpu().numpy()

model = NeuralNetwork().to(device=device)
#game = ConnectFour()


#state = game.get_board()

#policy, value = model.forward(state)
#print(forward)

#print(f'\n\nPolicy: {policy}\n Value:{value}')
