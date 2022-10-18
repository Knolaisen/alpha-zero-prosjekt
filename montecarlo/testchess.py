import chess
from chessboard import display
from MonteCarlo import MonteCarlo



board = chess.Board()
ai = MonteCarlo(15)
display.start(board.fen())
# print(board)
# print(board.legal_moves)
while True:
  if board.is_game_over():
    board.reset()
  #print()
  action = ai.Search(board)
  print(action)
  board.push(action)  # type: ignore
  display.start(board.fen())

  #print(action)

print(board.fullmove_number)
print(board.result())
while True:
  display.start(board.fen())