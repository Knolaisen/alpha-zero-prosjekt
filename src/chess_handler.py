from state import StateHandler
import chess
import numpy as np
import config as config  

class ChessStateHandler(StateHandler):
    def __init__(self, game: chess.Board()=None, to_play=1, turn=0):
        """
        Initialize the chess board
        """
        if game is None:
            self.board = chess.Board()
        else:
            self.board = game
        self.turn = turn
        self.to_play = to_play
        self.max_turns = config.MAX_TURNS

    def is_finished(self) -> bool:
        """
        Check if the game is finished (e.g. checkmate, stalemate, draw)
        Return True if the game is finished, False otherwise
        """
        if (len(self.get_legal_actions()) == 0): # TODO refactor
            return True
        if (self.max_turns < self.turn):
            return True
        return (self.board.is_variant_draw() or self.board.is_variant_loss() or self.board.is_variant_win())

    def get_winner(self) -> int:
        """
        Determine the winner of the game (-1 for black, 0 for draw, 1 for white)
        Return the winner as an integer
        """
        if (self.board.is_variant_draw):
            return 0
        elif(self.board.is_variant_loss & self.board.turn == "WHITE"):
            return -1
        elif(self.board.is_variant_loss & self.board.turn == "BLACK"):
            return 1
        else:
            assert "Game not finished!"

    def get_legal_actions(self) -> list:
        """
        Generates a list of legal moves for the current state of the game
        Return the legal moves as a list
        """
        moves_list = []
        for move in self.board.legal_moves:
            to_from_square = str(move.from_square) + str(move.to_square)
            moves_list.append((to_from_square, move))
        # sort the list of moves by the to_from_square
        moves_list.sort(key=lambda x: x[0])
        # return the list of moves without the to_from_square
        moves_list = [move[1] for move in moves_list]
        return moves_list
    
    def get_actions_mask(self) -> list:
        mask = np.zeros(len(config.ALL_POSSIBLE_MOVES))
        for legal_move in self.board.legal_moves:
            index = np.where(config.ALL_POSSIBLE_MOVES == str(legal_move)) 
            mask[index] = 1
        return mask

    def step(self, action) -> None:
        """
        Takes in a move and performs it on the board
        """
        self.board.push(action)
        self.to_play = -self.to_play
        self.turn += 1

    def step_back(self):
        """
        Takes a step back in the game
        """
        self.board.pop()
        self.to_play = -self.to_play
        self.turn -= 1


    def get_board_state(self) -> np.array:
        """
        Gives the current state of the board, used to visualize board, in the form of a numpy array
        """
        board = self.board
        board = str(board)
        board = board.replace("\n", "")
        board = board.replace(" ", "")

        # Define piece values
        piece_values = {
            "P": 1.0, "N": 2.0, "B": 3.0, "R": 4.0, "Q": 5.0, "K": 6.0,
            "p": -1.0, "n": -2.0, "b": -3.0, "r": -4.0, "q": -5.0, "k": -6.0
        }
        # Create an empty 8x8 NumPy array
        array = np.zeros((8, 8), dtype=float)
        # Iterate through each square on the board
        for i in range(64):
            # Get the piece at the current square
            piece = str(board)[i]
            # If there is a piece at the current square, update the array with the piece's value
            if piece == ".":
                continue
            else:
                rank, file = divmod(i, 8)  # Convert index to rank and file
                array[rank, file] = piece_values[piece]
        return array    
    
    def render(self):
        """
        Renders the board
        """
        print(self.board)

    def get_turn(self):
        """
        Returns the current turn
        """
        return self.turn
    
    def get_current_player(self):
        """
        Returns the player to play
        """
        return self.to_play
    
    def get_all_moves(self):
        """
        Returns all moves, regardless of being possible
        """
        return config.ALL_POSSIBLE_MOVES

    

if __name__ == "__main__":
    # Test the class
    state = ChessStateHandler()
    legal_actions = state.get_legal_actions()
    state.step(legal_actions[0])
    # state.step(legal_actions[1])
    print(state.get_board_state())
    print(state.get_legal_actions())