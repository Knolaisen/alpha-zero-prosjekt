import chess
from chessboard import display
from MonteCarlo import MonteCarlo



board = chess.Board()
ai = MonteCarlo(10)
display.start(board.fen())
# print(board)
# print(board.legal_moves)
while True:
  if board.is_game_over():
    board.reset()
  #print()
  action = ai.Search(board)
  print(f"{board.fullmove_number}. {action}")
  board.push(action)  # type: ignore
  display.start(board.fen())
  if board.fullmove_number % 3 == 0:
    # print(ai.stateDict)
    print(f"Size of dictionary: {len(ai.stateDict)}")
    moves = board.move_stack


print(board.fullmove_number)
print(board.result())
while True:
  display.start(board.fen())