from state import StateHandler
import chess
import numpy as np


class ChessStateHandler(StateHandler):
    def __init__(self, game: chess.Board()=None, to_play=1, turn=0):
        """Initialize the chess board """
        if game is None:
            self.board = chess.Board()
        else:
            self.board = game
        self.turn = turn
        self.to_play = to_play

    def is_finished(self) -> bool:
        """Check if the game is finished (e.g. checkmate, stalemate, draw)
        Return True if the game is finished, False otherwise"""
        return (self.board.is_variant_draw or self.board.is_variant_loss or self.board.is_variant_win)

    def get_winner(self) -> int:
        """Determine the winner of the game (-1 for black, 0 for draw, 1 for white)
        Return the winner as an integer"""
        if (self.board.is_variant_draw):
            return 0
        elif(self.board.is_variant_loss & self.board.turn == "WHITE"):
            return -1
        elif(self.board.is_variant_loss & self.board.turn == "BLACK"):
            return 1
        else:
            assert "Game not finished!!"

    def get_legal_actions(self) -> list:
        """Generates a list of legal moves for the current state of the game
        Return the legal moves as a list"""
        moves_list = []
        for i in self.board.legal_moves:
            moves_list.append(i)

        return moves_list

    def step(self, action) -> None:
        """Takes in a move and performs it on the board"""
        self.board.push(action)

    def step_back(self):
        """Takes a step back in the game"""
        self.board.pop()


    def get_board_state(self):
            """Gives the current state of the board, used to visualize board, in the form of a numpy array"""
            board = self.board
            board = str(board)
            board = board.replace("\n", "")
            board = board.replace(" ", "")

            # Define piece values
            piece_values = {
                "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
                "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6
            }
            # Create an empty 8x8 NumPy array
            array = np.zeros((8, 8), dtype=int)
            # Iterate through each square on the board
            print(board)
            for i in range(64):
                # Get the piece at the current square
                piece = str(board)[i]
                print(piece)
                # If there is a piece at the current square, update the array with the piece's value
                if piece == ".":
                    continue
                else:
                    rank, file = divmod(i, 8)  # Convert index to rank and file
                    array[rank, file] = piece_values[piece]
            return array    
    
    def render(self):
        """Renders the board"""
        print(self.board)

    def get_turn(self):
        """Returns the current turn"""
        return self.turn
    
    def get_current_player(self):
        """Returns the player to play"""
        return self.to_play
    

if __name__ == "__main__":
    # Test the class
    state = ChessStateHandler()
    legal_actions = state.get_legal_actions()
    state.step(legal_actions[0])
    # state.step(legal_actions[1])
    print(state.get_state())