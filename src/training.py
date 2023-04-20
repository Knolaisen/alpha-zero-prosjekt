from neural_network import NeuralNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from game_data import GameData
import config as config

# Hyperparameters


# 0) data processing
train_dataset: GameData
test_dataset: GameData
train_loader: DataLoader
test_loader: DataLoader

def updateDatasetAndLoad() -> None:
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
model: NeuralNet # TODO: Use the model neural_network.py

# 2) Loss and optimizer
criterion = nn.CrossEntropyLoss()  # TODO Finnes ulike CrossEntropyLoss - Sjekk ut det
optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

# 3) Training loop
def train_on_data() -> None:
    for epoch, (features, labels) in enumerate(train_loader):
        # Forward pass
        # Compute Loss
        loss = criterion(outputs, labels)
        # Backward pass
        # Update the model parameters

        # Print the loss every 100 iterations
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{epoch + 1}/{total_step}], Loss: {loss.item():.4f}')
    pass

# 4) Test the model
def test_model():
    pass

def train_ANET(iteration: int, simuations: int):
    """
    Trains the ANET model and saves the model to saved_models folder.
    Returns the trained model and the differnte trained versions of the model.
    """
    # 0) Setup, get device, node, game, etc.

    print("[DEVICE]: " + str(device))
    chess_game: ChessStateHandler # todo: Use the chess game
    # TODO: make it possible to cache the model

    episode: int
    for episode in range(config.EPISODES):
        print(f"Episode: {episode + 1} of {config.episodes}")
        
        # 1) Update the dataset
        updateDatasetAndLoad()
        # 2) Train the model
        train_on_data()
        # 3) Test the model
        test_model()
        # 4) Save the model
        torch.save(model.state_dict(), f"{config.MODEL_PATH}/model_{iteration}.pt")
        # 5) Save the model checkpoints
        # 6) Return the model and the different trained versions of the model
    
    return model

