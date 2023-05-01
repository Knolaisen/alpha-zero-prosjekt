import torch
from torch.utils.data import Dataset
import numpy as np
import config
import sys

np.set_printoptions(threshold=sys.maxsize) # For printing numpy arrays in full
class GameData(Dataset):
    def __init__(self):
        lines = GameData.read_data_file()

        self.features = lines[0]
        self.labels = lines[1]
        self.outcomes = lines[2]
        self.n_samples = len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], self.outcomes[index]

    def __len__(self):
        return self.n_samples

    @staticmethod
    def read_data_file() -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Read data file
        """
        with open(config.PERSONAL_FILE_NAME, "r") as f:
            lines = f.readlines()
            features = []
            labels = []
            outcomes = []
            for line in lines:
                feature, label, expected_outcome_probability = GameData._decode(line)
                features.append(feature)
                labels.append(label)
                outcomes.append(expected_outcome_probability)
            return features, labels, outcomes

    @staticmethod
    def clear_data_file() -> None:
        """
        Completely clears data file
        """
        with open(config.PERSONAL_FILE_NAME, "w") as f:
            f.truncate()

    @staticmethod
    def read_files_and_write_normalized_distributions() -> None:
        """
        Reads all files in the data folder and writes the normalized distributions to the main data file.
        """
        files: list[str] = GameData._get_files_in_data_folder()
        with open(config.MAIN_FILE_NAME, "r+") as f:
            lines = f.readlines()
            game_states = GameData._decode_all(lines)

            for file in files:
                with open(file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        feature, label = GameData._decode(line)
                        normalized_distribution = GameData._renormalize_list(label, feature)
                        GameData._replace_game_state_with_normalized_distribution()
                        
            f.truncate()
            # Must reset the file pointer to the beginning of the file after truncating it.
            f.seek(0)
            f.writelines(game_states)


    @staticmethod
    def _get_files_in_data_folder() -> "list[str]":
        """
        Returns a list of all files in the data folder
        """
        pass

    @staticmethod
    def _renormalize_list(original_list: list[float], sum_list: list[float]) -> list[float]:
        """
        Renormalizes the original list after adding all the values from the sum list.
        Both lists should have the same length.
    
        Args:
            original_list (list): The original list of values to be renormalized.
            sum_list (list): The list of values to add to the original list.
    
        Returns:
            list: The renormalized list.
        """
        # Step 1: Calculate the sum of the values in the sum list.
        sum_values = sum(sum_list)
    
        # Step 2: Check if the sum of the values is zero.
        if sum_values == 0:
            return original_list
    
        # Step 3: Calculate the renormalization factor.
        renorm_factor = sum(original_list) / sum_values
    
        # Step 4: Multiply each element in the original list by the renormalization factor.
        renorm_list = [x * renorm_factor for x in original_list]
    
        return renorm_list
    
    @staticmethod
    def _replace_game_state_with_normalized_distribution(game_state: np.ndarray, distribution: np.array) -> None:
        """
        Replaces the game state with the normalized distribution.
        """
        pass

    @staticmethod
    def add_data_to_replay_buffer(game_state: np.ndarray, distribution: np.array, expected_outcome_probability: float) -> None:
        """
        Adds more data saved file
        """

        GameData._add_data(game_state, distribution, expected_outcome_probability)

        with open(config.PERSONAL_FILE_NAME, "r+") as f:
            try:
                game_states = f.readlines()
            except:
                game_states = []

            amount_of_games, game_indices = GameData._count_games_and_indices(game_states)
            if amount_of_games > config.REPLAY_BUFFER_MAX_SIZE:
                # Remove the oldest game
                games_to_remove = amount_of_games - config.REPLAY_BUFFER_MAX_SIZE
                # -1  for zero index and another for the line before
                offset = game_indices[games_to_remove] - 2
                game_states = game_states[offset:]
                f.truncate(0)
                f.seek(0)
                f.writelines(game_states)

    @staticmethod
    def _count_games_and_indices(game_states) -> "int, list(int)":
        amount_of_games = 0
        game_start_indices = []
        current_index = 0

        game_state: str
        for game_state in game_states:
            if game_state.startswith(config.GAME_START_STATE):
                amount_of_games += 1
                game_start_indices.append(current_index)
            current_index += 1
        return amount_of_games, game_start_indices
    
    @staticmethod
    def _add_data(game_state: np.ndarray, distribution: np.array, expected_outcome_probability: float) -> None:
        """
        Adds the game state and distribution to the data file. All data is saved as a string in one line.
        """
        
        game_state.flatten()
        formatted_data: str = GameData._encode(game_state, distribution, expected_outcome_probability)

        with open(config.PERSONAL_FILE_NAME, "a+") as f:
            f.write(formatted_data + "\n")

    @staticmethod
    def _decode(line: str) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Decode a line of data
        """
        # Sanitize input
        line = GameData._sanitize(line)
        line = line.split(":")
        features = line[0]
        features = features.split(",")
        # Convert to float
        features = [float(num) for num in features]
        features = torch.from_numpy(np.array(features, dtype=np.float32))

        label = line[1]
        label = label.split(",")
        # Convert to float
        label = [float(num) for num in label]
        label = torch.from_numpy(np.array(label, dtype=np.float32))

        expected_outcome_probability = float(line[2])
        expected_outcome_probability = torch.from_numpy(np.array(expected_outcome_probability, dtype=np.float32))

        return features, label, expected_outcome_probability

    @staticmethod
    def _decode_all(lines: list[str]) -> "list[tuple[float, float, float]]":
        """
        Decode all lines of data
        """
        decoded_data = []
        for line in lines:
            decoded_data.append(GameData._decode_line(line))
        return decoded_data
    
    @staticmethod
    def _decode_line(line: str) -> "tuple[float, float, float]":
        """
        Decode a line of data
        """
        # Sanitize input
        line = GameData._sanitize(line)
        line = line.split(":")
        feature = line[0]
        feature = feature.split(",")

        label = line[1]
        label = label.split(",")
        # Convert to float
        feature = (float(num) for num in feature)
        label = (float(num) for num in label)
        expected_outcome_probability = float(line[2])

        return feature, label, expected_outcome_probability
    
    @staticmethod
    def _encode(features: np.array, label: np.array, expected_outcome_probability: float) -> np.array:
        """
        Encode a line of data
        format: features:label
        """
        line = str(features) + ":" + str(label) + ":" + str(expected_outcome_probability)
        return GameData._sanitize(line)

    @staticmethod
    def _sanitize(line: str) -> str:
        """
        Sanitize a line of data
        """
        line = line.replace("\n", "")
        line = line.replace(" ", ",")
        line = line.replace("]", "")
        line = line.replace("[", "")
        while ",," in line:
            line = line.replace(",,", ",")
        line = line.rstrip(",")
        line = line.lstrip(",")

        return line


if __name__ == "__main__":
    # print("======= Constructor GameData =======")
    # data = GameData()

    # features = np.asarray([-1.0, 1.0, 0.0, -1.0, -1.0])
    # label = np.asarray([0.0, 1.0, 0.0, 0.0])

    # board_state = "1.0,0.0,-1.0,-1.0:0.0,1.0,0.0,0.0"

    # print("======= ADD DATA =======")
    # # GameData._add_data(features, label)

    # print("======= ENCODE =======")
    # print(GameData._encode(features, label))
    # board_state = GameData._encode(features, label)

    # print("======= DECODE =======")
    # print(GameData._decode("-1, 0,-1,1,0:1.,0.,0.,0.,0."))

    # data = np.array(yourlist, dtype=np.float32)
    # GameData._add_data(data)
    GameData.clear_data_file()
