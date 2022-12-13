import torch
from torch import nn

# Eksempelkode som kjører på FasionMNIST dataset:
# URL: https://github.com/Knolaisen/alpha-zero-prosjekt/blob/Per-Ivar/examples/fashionMNIST.py

print(f'Using ResNet with double3x3 blocks (neuralNetwork_bigger.py)')
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
print(f"Using {device} device")

class BasicConvBlock(nn.Module):

    # Dimension after convolusion: n-f+2p+1, where n is the width and hight of a quadratic matrix, f is the kernel_size and p is the padding.
    # Including stride, the dimension becomes: 1 + (n-f+2p)/s

    def __init__(self, in_channels=2**8, out_channels=2**8) -> None:
        super(BasicConvBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False, device=device),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False, device=device),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()

    def forward(self, input):
        out = self.features(input)
        out += self.shortcut(out)
        out = nn.functional.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2**8):
        super(ResNet, self).__init__()        
        
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False, device=device),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ) 

        # 19 blocks high residual tower
        self.residual_tower = nn.Sequential(
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
            BasicConvBlock().to(device=device),
        )
    
        self.policy_head = nn.Sequential(
            nn.Conv2d(2**8, 2, kernel_size=1, stride=1, bias=False, device=device),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(0, -1),
            nn.Linear(7*6*2, 7, device=device),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(2**8, 1, kernel_size=1, stride=1, bias=False, device=device),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(0, -1),
            nn.Linear(7*6, 2**8, device=device),
            nn.ReLU(),
            nn.Linear(2**8, 1, device=device),
            nn.Tanh(),
        )

    def forward(self, input, tensor=False, only_policy=False):
        input = torch.from_numpy(input).to(device=device, dtype=torch.float)
        shape = 1, 1, 6, 7
        input = input.view(shape)
        #print(f'Input x: \n{input}, \n\tShape: {input.shape}')

        out = self.initial_block(input)
        out = self.residual_tower(out)
        
        #print(f'Residual tower output: {out}, with shape: \n\t{out.shape}')
        policy = self.policy_head(out)
        if only_policy:
            return policy.detach().cpu().numpy()
        value = self.value_head(out)

        if tensor: # returns tensor
            return policy, value

        # else returns numpy array
        return policy.detach().cpu().numpy(), value.detach().cpu().numpy()

#import environment
#game = environment.ConnectFour()
#game.move(column=4)
#game.switch_sides()
#game.move(column=3)
#game.switch_sides()
#state = game.get_board()
#print(f'Board state: \n{state}')


#network = BasicConvBlock(1, out_channels).to(device=device)
#testvalue = network.forward(state)
#print(f'Testvalue: {testvalue}, \n\tShape: {testvalue.shape}')

#resNet = ResNet().to(device=device)
#policy, value = resNet.forward(state)
#print(f'\n\nPolicy: {policy}\n Value:{value}')
