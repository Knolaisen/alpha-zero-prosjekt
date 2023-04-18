from state_handler.chess_handler import ChessStateHandler
from state import StateHandler

from neural_network.neural_network import NeuralNet, get_models
from MCTS import MCTS
import config
import torch


class TOPP:
    """
    The Tournament of Progressive Policies. This class is used to play a tournament of Hex between different models,
    and to keep track of the statistics of the tournament.
    """
    def __init__(self, models: list[NeuralNet]):
        self.models: list[NeuralNet] = models
        self.number_of_games = config.G
        self.statistics: dict[NeuralNet, int] = {}

        for model in models:
            self.statistics[model] = 0

    def play_tournament(self):
        for i in range(len(self.models)):
            modelOne: NeuralNet = self.models[i]
            opponent: NeuralNet
            for opponent in self.models[i + 1 :]:
                self.play_chess_games(modelOne, opponent)


    def play_chess_games(self, modelOne: NeuralNet, modelTwo: NeuralNet) -> None:
        """
        Play a game of Hex between two models
        """
        for game_nr in range(self.number_of_games):
            print(f"Game {game_nr} between {modelOne} and {modelTwo}")
            start_player = 1
            self._play_game(start_player, modelOne, modelTwo)
            
            # Let the other model start the game
            self._play_game(start_player, modelTwo, modelOne)

    def _play_game(
        self, start_player: int, modelOne: NeuralNet, modelTwo: NeuralNet
    ) -> None:
        """
        Play a game of Hex between two models and update the statistics
        """
        board = ChessStateHandler(self.board_size, start_player)
        while not board.is_finished():
            player: float = board.get_current_player()

            move: tuple[int, int]
            if player == start_player:
                move = MCTS.defaultPolicy(board, modelOne, 0)
            else:
                move = MCTS.defaultPolicy(board, modelTwo, 0)
            board.step(move)

        winner = board.get_winner()
        if start_player == winner:
            self.statistics[modelOne] += 1
        else:
            self.statistics[modelTwo] += 1


    def get_results(self) -> dict[NeuralNet, int]:
        return self.statistics

