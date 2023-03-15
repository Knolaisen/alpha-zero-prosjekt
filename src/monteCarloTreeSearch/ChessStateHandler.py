from src.monteCarloTreeSearch.state import state_handler
import chess

class ChessState(state_handler):
	def __init__(self):
		# Initialize the chess board here
		# You can use a library like python-chess to represent the board and handle moves
		self.Board = chess.Board()
		
	def is_finished(self) -> bool:
		# Check if the game is finished (e.g. checkmate, stalemate, draw)
		# Return True if the game is finished, False otherwise
		if (self.Board.is_variant_draw or self.Board.is_variant_loss or self.Board.is_variant_win):
			return True
		else:
			return False 

	def get_winner(self) -> int:
		# Determine the winner of the game (-1 for black, 0 for draw, 1 for white)
		# Return the winner as an integer
		if (self.Board.is_variant_draw):
			return 0
		elif(self.Board.is_variant_loss & self.Board.turn == "WHITE"):
			return -1
		elif(self.Board.is_variant_loss & self.Board.turn == "BLACK"): 
			return 1

	def get_legal_actions(self) -> list:
		# Get a list of legal moves for the current state of the game
		# You can use a library like python-chess to generate legal moves
		# Return the legal moves as a list
		moves_list = []
		for i in self.Board.legal_moves:
			moves_list.append(i)

		return moves_list

	def move(self, action) -> None:
		# Execute a move on the chess board
		# You can use a library like python-chess to execute the move
		self.Board.push(action)

	def get_state(self):
		# Get the current state of the chess board
		# You can use a library like python-chess to represent the board state
		# Return the board state
		return self.Board