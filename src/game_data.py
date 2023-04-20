import torch
from torch.utils.data import Dataset
import numpy as np
import config
import sys
file_name = config.MCTS_DATA_PATH + "/data_file.csv"

np.set_printoptions(threshold=sys.maxsize)
class GameData(Dataset):
    def __init__(self):
        lines = GameData.read_data_file()

        self.features = lines[0]
        self.labels = lines[1]
        self.n_samples = len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.n_samples

    @staticmethod
    def read_data_file() -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Read data file
        """
        with open(file_name, "r") as f:
            lines = f.readlines()
            features = []
            labels = []
            for line in lines:
                feature, label = GameData._decode(line)
                features.append(feature)
                labels.append(label)
            return features, labels

    @staticmethod
    def clear_data_file() -> None:
        """
        Completely clears data file
        """
        with open(file_name, "w") as f:
            f.truncate()

    @staticmethod
    def add_data(game_state: np.ndarray, distribution: np.array) -> None:
        """
        Tar inn np array og lagrer tilstand til brettet 
        Adds more data saved file 

        """
        
        game_state.flatten()
        formatted_data: str = GameData._encode(game_state, distribution)

        with open(file_name, "a") as f:
            f.write(formatted_data + "\n")

    @staticmethod
    def _decode(line: str) -> "tuple[torch.Tensor, torch.Tensor]":
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
        return features, label

    @staticmethod
    def _encode(features: np.array, label: np.array) -> np.array:
        """
        Encode a line of data
        format: features:label
        """
        line = str(features) + ":" + str(label)
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
    # # GameData.add_data(features, label)

    # print("======= ENCODE =======")
    # print(GameData._encode(features, label))
    # board_state = GameData._encode(features, label)

    # print("======= DECODE =======")
    # print(GameData._decode("-1, 0,-1,1,0:1.,0.,0.,0.,0."))

    # data = np.array(yourlist, dtype=np.float32)
    # GameData.add_data(data)
    GameData.clear_data_file()
