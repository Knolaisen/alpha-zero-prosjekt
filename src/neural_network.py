
import torch
import torch.nn as nn
import numpy as np
import config as config
from state import StateHandler

class NeuralNet(nn.Module):
    def __init__(self, num_residual_blocks=config.NUM_RESIDUAL_BLOCKS, num_filters=config.NUM_FILTERS):
        super(NeuralNet, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(config.INPUT_SIZE, num_filters, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(*[ResidualBlock(num_filters, num_filters, kernel_size=3, padding=1) for _ in range(num_residual_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 65 * 1, config.OUTPUT_SIZE),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(config.INPUT_SIZE , 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        x = self.residual_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    
    def get_best_move_index(self, state: torch.Tensor, game: StateHandler) -> int:
        """Get the best move from the model"""
        # Disable gradient calculation to speed up the process
        with torch.no_grad():
            print("Getting best move")
            print("State: ", state)
            output: torch.Tensor = self(state)
            
            # Convert to numpy array and by correct device
            np_arr: np.array
            if config.DEVICE == "cuda":
                np_arr: np.array = output.detach().cuda().numpy().astype(np.float32)
            else:
                np_arr: np.array = output.detach().cpu().numpy().astype(np.float32)

            # Compute softmax
            result = NeuralNet.softmax(np_arr)
            # Mask the array with the actions that are not allowed
            np_arr = result * game.get_actions_mask()
            # Get the index of the best move
            index = np.argmax(np_arr)
            return index
    
    def softmax(x: torch.Tensor ) -> np.array:
        # Subtract the maximum value to avoid numerical issues with large exponents
        e_x = np.exp(x - np.max(x))

        # Compute softmax values
        return e_x / np.sum(e_x, axis=0)

    
    def default_policy(self, game: StateHandler) -> any:
        """
        Default policy finds the best move from the model
        """

        state = game.get_board_state()
        state = state.flatten()
        # Add the turn indicator
        state = np.insert(state, 0, game.get_current_player())
        # state = np.expand_dims(state, axis=0)  # Add the batch dimension
        # state = np.expand_dims(state, axis=(0,1))  # Add the batch dimension

        state: torch.Tensor = torch.from_numpy(state).float().to(config.DEVICE)
        print("State: ", state)
        # Find the correct move from the legal games and return it
        # Use the mask and legal moves to find the correct move
        legal_moves = game.get_legal_actions()
        mask = game.get_actions_mask()
        best_move_index = self.get_best_move_index(state, game)
        
        # best_move_index = 5, [0,1,0,0,0,1,1]
        # legal_moves = [1, 2, 3]

        converted_index = 0
        for i in range(best_move_index):
            if mask[i] == 1:
                converted_index += 1

        best_move = legal_moves[converted_index]
        return best_move

    def save_model(self, iteration: int, simulations: int) -> None:
        """Save the model to a file"""
        file_name = f"model_{iteration}_{simulations}.pt"
        torch.save(self.state_dict(), f"{config.MODEL_PATH}/{file_name}")

    @staticmethod
    def load_model(file_name: str) -> "NeuralNet":
        """Load the model from a file"""

        model = NeuralNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)

        loaded_model: "NeuralNet"= torch.load(f"{config.MODEL_PATH}/{file_name}")

        model.load_state_dict(loaded_model.state_dict())

        model = model.eval()

        return model

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class TestNet(nn.Module):
    def __init__(self) -> None:
        super(TestNet, self).__init__()
        self.input_layer = nn.Linear(65, 256)
        self.hidden_layer = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 64)
    
    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x



if __name__ == "__main__":
    # model = NeuralNet()
    # x = torch.randn(65)
    # out = model(x)
    # print(out)



    array_1d = np.random.randn(65)
    # Reshape the 1D array into a 4D tensor with shape (1, 1, 65, 1):
    # array_4d = array_1d.reshape(256, 65, 1, 1)
    # Foremat when reshape: (batch_size, num_channels, height, width)
    array_3d = np.stack([array_1d] * 65, axis=0)
    array_4d = array_3d.reshape(1, 65, 65, 1)
    # Convert the 4D NumPy array to a PyTorch tensor:
    input_tensor = torch.from_numpy(array_4d).float()
    # Create an instance of the NeuralNet class and pass the input tensor to it:
    model = NeuralNet()
    output = model(input_tensor)
    print(output)
