import numpy as np
from env import ConnectFour

game = ConnectFour()

print(f"Initial board\n {game.get_board()}\n")

print("Do 6 moves in column 5")
for i in range(6):
  player = 1 if i % 2 == 0 else -1
  game.move(5, player)

print(game.get_board())

print("\nGet available moves")
print(game.get_available_moves())

print("\nIs game in win state?")
print(game.is_win_state())

print("\nPlay 4 consecutive moves in column 0")
for i in range(4):
  game.move(0, 1)
print(game.get_board())

print("\nIs game in win state?")
print(game.is_win_state())

print("\nReset board")
game = ConnectFour()
print(game.get_board())

print("\nPlay 4 moves in bottom row")
for i in range(1, 5):
  game.move(i, -1)

print("\nIs game in win state?")
print(game.is_win_state())