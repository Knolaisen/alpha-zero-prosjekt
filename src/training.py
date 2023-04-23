import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from game_data import GameData
from neural_network import NeuralNet, transform_2d_to_tensor
from mcts import *

from node import Node
import config
import copy
from topp import TOPP
import cProfile
from pstats import SortKey
import pstats


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
    train_dataset = GameData()
    test_dataset = GameData()
    train_loader = DataLoader(dataset=train_dataset)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE)


# 1) Model

model = NeuralNet().to(config.DEVICE)

# 2) Loss and optimizer

criterion = nn.CrossEntropyLoss()  # Finnes ulike CrossEntropyLoss - Sjekk ut det
# There are many options of optimizers: Adagrad, Stochastic Gradient Descent (SGD), RMSProp, and Adam.
# We should try out different optimizers and see which one works best.
optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)


def train_on_data(game: StateHandler):
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        print(f"features.shape = {features.shape}")
        # forward pass
        
        output = model(transform_2d_to_tensor(game, features))
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


def train_ANET(iteration: int, simulations: int):
    """
    Trains the ANET model and saves the model to saved_models folder.
    Returns the trained model and the differnte trained versions of the model.
    """
    print("[DEVICE]: " + str(config.DEVICE))
    game = ChessStateHandler()
    root = Node(game)

    cached_models = []
    for i in range(config.EPISODES):
        print(f"Iteration: {i + 1} of {config.EPISODES}")
        GameData.clear_data_file()
        generate_test_data(
            root, iteration, simulations, model
        )  # TODO Model not defined in train_ANET
        updateDatasetAndLoad()
        train_on_data(root.get_state())
        # with torch.no_grad():
        # testModel(model)

        if i % (config.EPISODES // (config.M - 1)) == 0 or (i == (config.EPISODES - 1)):
            # if i % (config.episodes // (config.M - 2)) == 0:
            print("[INFO] Saving model...")
            model.saveModel(i, simulations)
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
        model, 7, 100, 500).to(get_default_config.DEVICE())

    test_case = torch.Tensor([1, 0, -1, 1, 0]).to(get_default_config.DEVICE())

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

