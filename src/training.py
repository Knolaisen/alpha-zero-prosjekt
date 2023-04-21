import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2

import numpy as np
import matplotlib.pyplot as plt
from game_data import GameData
from neural_network import NeuralNet
from mcts import *

from node import Node
import config
import copy
from topp import TOPP
import cProfile
from pstats import SortKey
import pstats

# GPU config


def get_default_device() -> torch.device:
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# def train_model(model, train_loader, test_loader, loss_fn, optimizer, n_epochs=1):
device = get_default_device()


# Hyperparameters

player_id_len = 1
hidden_size = 128
num_epochs = 10
batch_size = 4  # spørsmål?
learning_rate = config.LEARNING_RATE

# 0) Data processing
train_dataset: GameData
test_dataset: GameData
train_loader: DataLoader
test_loader: DataLoader


def updateDatasetAndLoad():
    """
    Updates the data file with the new data
    """
    global train_dataset
    global test_dataset
    global train_loader
    global test_loader
    train_dataset = GameData(train=True)
    test_dataset = GameData(train=False)
    train_loader = DataLoader(dataset=train_dataset)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)


# 1) Model

model = NeuralNet(
    # config.INPUT_SIZE,
    # config.HIDDEN_SIZE,
    # # config.hidden_size1,
    # # config.hidden_size2,
    # # config.hidden_size3,
    # config.OUTPUT_SIZE,
).to(device)

# 2) Loss and optimizer

criterion = nn.CrossEntropyLoss()  # Finnes ulike CrossEntropyLoss - Sjekk ut det
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def trainOnData():
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        # forward pass
        output = model(features)
        loss = criterion(output, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print status
        if (i + 1) % 1 == 0:
            pass

    print(f"loss = {loss.item(): .4f}")
    config.epsilon *= config.epsilon_decay


def train_ANET(iteration: int, simuations: int):
    """
    Trains the ANET model and saves the model to saved_models folder.
    Returns the trained model and the differnte trained versions of the model.
    """

    device = get_default_device()
    print("[DEVICE]: " + str(device))
    hexGame = ChessStateHandler()
    root = Node(hexGame)

    cached_models = []
    for i in range(config.EPISODES):
        print(f"Iteration: {i + 1} of {config.EPISODES}")
        GameData.clear_data_file()
        generate_test_data(
            root, iteration, simuations, model
        )  # TODO Model not defined in train_ANET
        updateDatasetAndLoad()
        trainOnData()
        # with torch.no_grad():
        # testModel(model)

        if i % (config.EPISODES // (config.M - 1)) == 0 or (i == (config.EPISODES - 1)):
            # if i % (config.episodes // (config.M - 2)) == 0:
            print("[INFO] Saving model...")
            model.saveModel(i, simuations)
            batch_model = copy.deepcopy(model)
            cached_models.append(batch_model)
            print("[INFO] Model saved!")

    print("[INFO] Training complete!")

    return cached_models


# # 4) Visualize
def initializeWriter():
    writer = SummaryWriter("runs/Hex")  # Maybe use logs folder instead?
    return writer


# Create a hexagonal board

def main():
    models = train_ANET(config.MCTS_GAMES, config.MCTS_SIMULATIONS)
    print("[INFO] Check the saved networks with loaded ones...")

    print("[INFO] Running TOPP...")
    tournament: TOPP = TOPP(models)
    # Run TOPP
    tournament.play_tournament()
    print("[INFO] TOPP complete!")
    tournament.print_results()


if __name__ == "__main__":
    """
    model = NeuralNet(50, 128, 49)
    trained_model = NeuralNet.loadModel(
        model, 7, 100, 500).to(get_default_device())

    test_case = torch.Tensor([1, 0, -1, 1, 0]).to(get_default_device())

    result = trained_model(test_case)

    print("Test case 2: " + str(result))


    """

    cProfile.run("main()", "output.dat")

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()


# from neural_network import NeuralNet
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import numpy as np

# from game_data import GameData
# import config as config

# # Hyperparameters


# # 0) data processing
# train_dataset: GameData
# test_dataset: GameData
# train_loader: DataLoader
# test_loader: DataLoader

# def updateDatasetAndLoad() -> None:
#     """
#     Updates the data file with the new data
#     """
#     global train_dataset
#     global test_dataset
#     global train_loader
#     global test_loader
#     train_dataset = GameData()
#     test_dataset = GameData()
#     train_loader = DataLoader(dataset=train_dataset)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE)

# # 1) Model
# model: NeuralNet # TODO: Use the model neural_network.py

# # 2) Loss and optimizer
# criterion = nn.CrossEntropyLoss()  # TODO Finnes ulike CrossEntropyLoss - Sjekk ut det
# optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

# # 3) Training loop
# def train_on_data() -> None:
#     for epoch, (features, labels) in enumerate(train_loader):
#         # Forward pass
#         # Compute Loss
#         loss = criterion(outputs, labels)
#         # Backward pass
#         # Update the model parameters

#         # Print the loss every 100 iterations
#         if (epoch + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{epoch + 1}/{total_step}], Loss: {loss.item():.4f}')
#     pass

# # 4) Test the model
# def test_model():
#     pass

# def train_ANET(iteration: int, simuations: int):
#     """
#     Trains the ANET model and saves the model to saved_models folder.
#     Returns the trained model and the differnte trained versions of the model.
#     """
#     # 0) Setup, get device, node, game, etc.

#     print("[DEVICE]: " + str(device))
#     chess_game: ChessStateHandler # todo: Use the chess game
#     # TODO: make it possible to cache the model

#     episode: int
#     for episode in range(config.EPISODES):
#         print(f"Episode: {episode + 1} of {config.episodes}")
        
#         # 1) Update the dataset
#         updateDatasetAndLoad()
#         # 2) Train the model
#         train_on_data()
#         # 3) Test the model
#         test_model()
#         # 4) Save the model
#         torch.save(model.state_dict(), f"{config.MODEL_PATH}/model_{iteration}.pt")
#         # 5) Save the model checkpoints
#         # 6) Return the model and the different trained versions of the model
    
#     return model

