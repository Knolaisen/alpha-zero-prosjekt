import torch
import numpy as np
# ================= Configurations =================



# ============== Training  parameters ==============
G = 2  # Number of games between opponents in TOPP
M = 5  # Number of model versions to cache 
EPISODES = 20  # Number of episodes to train ANET for 
LEARNING_RATE = 0.05  # Learning rate
SIMULATIONS = 50  # MCTS rollout games
MCTS_GAMES = 20  # Number of MCTS games
TIME_LIMIT = 10  # Time limit for MCTS
BATCH_SIZE = 32  # Batch size for training
NUM_EPOCHS = 10  # Number of epochs to train for
EPSILON = 0.3 # Epsilon for epsilon greedy

# ================= Hardwaresettings =================
DEVICE: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ====================== Paths =======================

MCTS_DATA_PATH = "src/saved_mcts/"# Path to the folder containing the data
MODEL_PATH = "src/saved_models/"# Path to the folder containing the models


# ===================== Neural Network settings =====================
INPUT_SIZE = 12*8*8  # Input size
HIDDEN_SIZE = 2**10  # Hidden size
OUTPUT_SIZE = 12*8*8  # Output size
# ====================== Chess Values =======================
squares = np.array(["a1","a2","a3","a4","a5","a6","a7","a8",
                    "b1","b2","b3","b4","b5","b6","b7","b8",
                    "c1","c2","c3","c4","c5","c6","c7","c8",
                    "d1","d2","d3","d4","d5","d6","d7","d8",
                    "e1","e2","e3","e4","e5","e6","e7","e8",
                    "f1","f2","f3","f4","f5","f6","f7","f8",
                    "g1","g2","g3","g4","g5","g6","g7","g8",
                    "h1","h2","h3","h4","h5","h6","h7","h8"])

allmoves = np.zeros((64, 64), dtype=object)
for i in range(64):
    for j in range(64):
        if i != j:
            allmoves[i][j] = squares[i] + squares[j]

        # Remove diagonal elements entirely
mask = np.eye(64, dtype=bool)
allmoves = allmoves[~mask].reshape(64, 63)
ALL_POSSIBLE_MOVES = allmoves
    