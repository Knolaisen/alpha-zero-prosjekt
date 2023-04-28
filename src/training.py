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

sys.setrecursionlimit(5000)

# 0) Data processing
train_dataset: GameData
test_dataset: GameData
train_loader: DataLoader
test_loader: DataLoader


def update_dataset_and_load():
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

policy_loss = nn.CrossEntropyLoss()
value_loss = nn.MSELoss()

def alpha_zero_loss(policy_pred: torch.Tensor, MCTS_policy_prob: torch.Tensor, value_pred: torch.Tensor, MCTS_value: torch.Tensor):
#    print("policy_pred: ", policy_pred, "MCTS_policy_prob: ", MCTS_policy_prob, "value_pred: ", value_pred, "MCTS_value: ", MCTS_value)
    # print("policy_pred: ", policy_pred.shape, "MCTS_policy_prob: ", MCTS_policy_prob.shape, "value_pred: ", value_pred.shape, "MCTS_value: ", MCTS_value.shape)
    return policy_loss(policy_pred, MCTS_policy_prob) + value_loss(value_pred, MCTS_value[0])


# criterion = nn.CrossEntropyLoss()  # Finnes ulike CrossEntropyLoss - Sjekk ut det
# There are many options of optimizers: Adagrad, Stochastic Gradient Descent (SGD), RMSProp, and Adam.
# We should try out different optimizers and see which one works best.
optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)


def train_on_data():
    '''
    Trains the model on the data in the data file 
    '''
    for epoch, (features, labels, expected_outcome_probability) in enumerate(train_loader):
        features = features.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        # forward pass
        MCTS_policy_prob, MCTS_value = model(transform_2d_to_tensor(features=features))

        # loss = alpha_zero_loss(MCTS_policy_prob, features, MCTS_value, )¨
        # print("expected_outcome_probability: " + str(expected_outcome_probability) + " MCTS_value: " + str(MCTS_value))
        loss = alpha_zero_loss(labels, MCTS_policy_prob, expected_outcome_probability, MCTS_value)

        # loss = criterion(output[0], labels)

        # backward
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()

    print(f"{epoch} loss = {loss.item(): .4f}")



def train_ANET(iteration: int, rounds: int):
    """
    Trains the ANET model and saves the model to saved_models folder.
    Returns models cached during training.
    """
    print("[DEVICE]: " + str(config.DEVICE))
    game = ChessStateHandler()
    root = Node(game)

    cached_models = []
    for i in range(config.EPISODES):
        print(f"Iteration: {i + 1} of {config.EPISODES}")
        generate_test_data(root, iteration, rounds, model) 
        update_dataset_and_load()
        
        train_on_data()
        # with torch.no_grad():
        # testModel(model)

        if i % (config.EPISODES // (config.M - 1)) == 0 or (i == (config.EPISODES - 1)):
            # if i % (config.episodes // (config.M - 2)) == 0:
            print("[INFO] Saving model...")
            model.save_model(i, rounds, config.NUM_RESIDUAL_BLOCKS, config.NUM_FILTERS)
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
    # cProfile.run("main()", "output.dat")

    # with open("output_time.txt", "w") as f:
    #     p = pstats.Stats("output.dat", stream=f)
    #     p.sort_stats("time").print_stats()

    # with open("output_calls.txt", "w") as f:
    #     p = pstats.Stats("output.dat", stream=f)
    #     p.sort_stats("calls").print_stats()


    # 
    update_dataset_and_load()
    train_on_data()
