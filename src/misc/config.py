import torch
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


