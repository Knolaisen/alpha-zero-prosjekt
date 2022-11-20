import numpy as np
import random
from env import ConnectFour

for i in range(10):
    game = ConnectFour()
    player = 1

    while not game.is_game_over():
        move = random.choice(game.get_available_moves())
        game.move(move, player)
        player *= -1

    winner, player = game.is_win_state()
    print("\nGame is over")
    if winner:
        print(f"Winner is {player}")
        print(game.get_board())
    else:
        print("Game is draw")
        print(game.get_board())
