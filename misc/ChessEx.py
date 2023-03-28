import random
import chess
import numpy as np
'''
This is an example chessboard we used in the early stages.
'''
board = chess.Board()

board.legal_moves  

# making a list of legal moves

print(board.legal_moves)
moves_list = []
index=0
for i in board.legal_moves:
    print(i)
    moves_list.append(i)

# how to add move from legal moves list
# print(board)
# board.push(moves_list[0])
# print(board)

board.push_san("e4")
board.push_san("e5")
board.push_san("Qh5")
board.push_san("Nc6")
board.push_san("Bc4")
board.push_san("Nf6")
board.push_san("Qxf7")
print(board.is_checkmate())
print(board)
print(board.legal_moves)



