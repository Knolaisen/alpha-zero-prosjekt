import random
import chess
from chessboard import display
from MonteCarlo import MonteCarlo



board = chess.Board()
ai = MonteCarlo(20)
display.start(board.fen())
print(board)
print(board.legal_moves)
while input() != " ":
  if board.is_game_over():
    board.reset()
  #print()
  action = ai.Search(board)
  board.push(action)
  display.start(board.fen())

  #print(action)

print(board.fullmove_number)
print(board.result())
while True:
  display.start(board.fen())