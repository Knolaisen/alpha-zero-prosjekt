import random
import chess
from chessboard import display

'''
This is an example chessboard we used in the early stages.
'''
board = chess.Board()
display.start(board.fen())
print(board)
print(board.legal_moves)
while not board.is_game_over():
  #print()
  action = random.choice(list(board.legal_moves))
  board.push(action)
  display.start(board.fen())
  #print(action)

print(board.fullmove_number)
print(board.result())
while True:
  display.start(board.fen())