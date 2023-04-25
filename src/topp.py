from chess_handler import ChessStateHandler
from state import StateHandler

from neural_network import NeuralNet
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
        self.print_results()


    def play_chess_games(self, modelOne: NeuralNet, modelTwo: NeuralNet) -> None:
        """
        Play a game of Hex between two models
        """
        for game_nr in range(self.number_of_games):
            print(f"Game {game_nr} between model_{modelOne.module_iterations_trained} and model_{modelTwo.module_iterations_trained}")
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
        board = ChessStateHandler()
        while not board.is_finished():
            player: float = board.get_current_player()

            move: tuple[int, int]
            if player == start_player:
                move = modelOne.default_policy(board)
            else:
                move = modelTwo.default_policy(board)
            board.step(move)

        winner = board.get_winner()
        if start_player == winner:
            self.statistics[modelOne] += 1
        else:
            self.statistics[modelTwo] += 1

    def play_vs_bot(self):
        model = self.models[-1]
        game = ChessStateHandler()

        while not game.is_finished():
            game.render()
            current_player = game.get_current_player()
            if current_player == 1:
                user_move = self.user_input()
                game.step(user_move)

            else:
                move = model.default_policy(game)
                game.step(move)
        game.render()
        winner = game.get_winner()
        if winner == 1:
            print("Congratulations, you won!")
        else:
            print("You lost to the bot!")

    def user_input(self) -> str:
        while True:
            user_input = input("Your turn to place move: Example (a1a2) ")
            return user_input

    def get_results(self) -> dict[NeuralNet, int]:
        return self.statistics
    
    def print_results(self):
        for i in range(len(self.models)):
            print(
                f"Model_{self.models[i].module_iterations_trained} won {self.statistics.get(self.models[i])} times"
            )

