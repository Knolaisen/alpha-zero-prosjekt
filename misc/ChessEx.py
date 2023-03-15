import random
import chess
'''
This is an example chessboard we used in the early stages.
'''
board = chess.Board()

board.legal_moves  
moves_list = [] 
board.push_san("e4")
board.push_san("e5")
board.push_san("Qh5")
board.push_san("Nc6")
print(board.legal_moves)
for i in board.legal_moves:
    moves_list.append(i)
print(moves_list)


board.push_san("Bc4")
board.push_san("Nf6")
board.push_san("Qxf7")
print(board.is_checkmate())
print(board)
print(board.legal_moves)



