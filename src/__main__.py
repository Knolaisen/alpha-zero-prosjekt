import sys
import config
from topp import TOPP
from training import train_ANET
from torch import torch
from neural_network import NeuralNet
import cProfile
import pstats


def main():
    sys.setrecursionlimit(5000)

    models = train_ANET(config.MCTS_GAMES, config.MCTS_SIMULATIONS)
    print("[INFO] Running TOPP...")
    tournament: TOPP = TOPP(models)
    # Run TOPP
    tournament.play_tournament()
    print("[INFO] TOPP complete!")
    print()

    # # Fight between the worst and the best and some random
    # print("[INFO] Loading long trained models ...")
    # loaded_models = NeuralNet.get_models(4, 5)
    # print("[INFO] Running TOPP with loaded models...")
    # tournament: TOPP = TOPP(loaded_models)
    # # Run TOPP
    # tournament.play_tournament()

    print()
    user_input = input("Do you wanna play the best model? Press [enter] to continue")
    tournament.play_vs_bot()

if __name__ == "__main__":
    cProfile.run("main()", "output.dat")

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()