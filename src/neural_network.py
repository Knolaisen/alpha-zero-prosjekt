
import torch
import torch.nn as nn
import numpy as np
import config as config
from state import StateHandler


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
    
    def get_best_move_index(self, state: torch.Tensor, game: StateHandler) -> int:
        """Get the best move from the model"""
        # Disable gradient calculation to speed up the process
        with torch.no_grad():
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
    
    def softmax(x) -> np.array:
        # Subtract the maximum value to avoid numerical issues with large exponents
        e_x = np.exp(x - np.max(x))

        # Compute softmax values
        return e_x / np.sum(e_x, axis=0)

    
    def default_policy(self, game: StateHandler) -> any:
        """
        Default policy finds the best move from the model
        """

        state = game.get_board_state()
        state: torch.Tensor = torch.from_numpy(state).to(config.DEVICE)
        
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
