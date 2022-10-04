import chess
import random
from chessboard import display

board = chess.Board()

while not board.is_game_over():
  display.start(board.fen())
  action = random.choice(list(board.legal_moves))
  board.push(action)
  print(f"{board.fullmove_number}. {action}")

print(board.result())

while True:
  display.start(board.fen())
