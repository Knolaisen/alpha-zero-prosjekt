from src.monteCarloTreeSearch.state import state_handler


class ChessState(state_handler):
	def __init__(self):
		# Initialize the chess board here
		# You can use a library like python-chess to represent the board and handle moves
		pass
		
	def is_finished(self) -> bool:
		# Check if the game is finished (e.g. checkmate, stalemate, draw)
		# Return True if the game is finished, False otherwise
		pass

	def get_winner(self) -> int:
		# Determine the winner of the game (-1 for black, 0 for draw, 1 for white)
		# Return the winner as an integer
		pass

	def get_legal_actions(self) -> list:
		# Get a list of legal moves for the current state of the game
		# You can use a library like python-chess to generate legal moves
		# Return the legal moves as a list
		pass

	def move(self, action) -> None:
		# Execute a move on the chess board
		# You can use a library like python-chess to execute the move
		pass

	def get_state(self):
		# Get the current state of the chess board
		# You can use a library like python-chess to represent the board state
		# Return the board state
		pass