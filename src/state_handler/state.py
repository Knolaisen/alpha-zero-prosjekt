from abc import ABC, abstractmethod


class StateHandler(ABC):

    def move_to_state(self, game, to_play=1, turn=0):
        self.to_play = to_play
        self.turn = turn
        self.game = game
        return self
        

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
    def get_legal_actions_mask(self) -> list:
        """
        Get a list of legal action at current game state
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
    def get_board_state(self):
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
