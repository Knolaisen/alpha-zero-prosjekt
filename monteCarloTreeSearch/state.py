from abc import ABC, abstractmethod
from Action import Action

class state_handler(ABC):
    
	@abstractmethod
	def is_finished(self) -> bool:
		"""
		When the game is done is_finished will be true.
		"""
		pass

	@abstractmethod
	def get_winner(self) -> int:
		"""
		Whom won this game
		"""
		pass

	@abstractmethod
	def get_legal_actions(self) -> list:
		"""
		Get a list of legal action at current game state
		"""
		pass

	@abstractmethod
	def move(self, action: Action) -> None:
		"""
		Do the move in the game
		"""
		pass
	
	@abstractmethod
	def get_state(self):
		"""
		Get the state of the game.
		"""
		pass
	

