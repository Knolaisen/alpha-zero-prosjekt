from abc import ABC, abstractmethod
import numpy as np


class StateHandler(ABC):
    @abstractmethod
    def is_finished(self) -> bool:
        """
        When the game is done is_finished will be true.
        """
        pass

    @abstractmethod
    def get_winner(self) -> int:
        """
        Whom won this game -1, 0, 1. 1 is the first player, -1 is the second player and 0 is a draw.
        """
        pass

    @abstractmethod
    def get_legal_actions(self) -> list:
        """
        Get a list of legal action at current game state
        """
        pass

    @abstractmethod
    def get_actions_mask(self) -> list:
        """
        Get a list of legal action at current game state.
        Available moves as 1, Illegal 0. 
        [0,0,1,1,0,1] <- Chessboard
        """
        pass

    @abstractmethod
    def step(self, action) -> None:
        """
        Do the move in the game
        """
        pass
    
    @abstractmethod
    def step_back(self) -> None:
        """
        Take a step back in the game
        """
        pass

    @abstractmethod
    def get_board_state(self) -> np.array:
        """
        Get the state of the game.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Render the game state
        """
        pass

    @abstractmethod
    def get_turn(self):
        """
        Get the current turn
        """
        pass
    
    @abstractmethod
    def get_current_player(self):
        """
        Get the current player
        """
        pass
    @abstractmethod
    def render(self) -> None:
        """
        Render the game state
        """
        pass
