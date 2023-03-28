from src.monteCarloTreeSearch.state import state_handler
import chess

class ChessState(state_handler):
	def __init__(self):
		"""Initialize the chess board """
		self.Board = chess.Board()
		
	def is_finished(self) -> bool:
		"""Check if the game is finished (e.g. checkmate, stalemate, draw)
		Return True if the game is finished, False otherwise"""
		if (self.Board.is_variant_draw or self.Board.is_variant_loss or self.Board.is_variant_win):
			return True
		else:
			return False 

	def get_winner(self) -> int:
		"""Determine the winner of the game (-1 for black, 0 for draw, 1 for white)
		Return the winner as an integer"""
		if (self.Board.is_variant_draw):
			return 0
		elif(self.Board.is_variant_loss & self.Board.turn == "WHITE"):
			return -1
		elif(self.Board.is_variant_loss & self.Board.turn == "BLACK"): 
			return 1
		else:
			assert "Game not finished!!"

	def get_legal_actions(self) -> list:
		"""Generates a list of legal moves for the current state of the game
		Return the legal moves as a list"""
		moves_list = []
		for i in self.Board.legal_moves:
			moves_list.append(i)

		return moves_list

	def move(self, action) -> None:
		"""Takes in a move and performs it on the board"""

		self.Board.push(action)

	def get_state(self):
		"""Gives the current state of the board, uset to visualize board"""
		return self.Board