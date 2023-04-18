from state import StateHandler
import chess


class ChessStateHandler(StateHandler):
    def __init__(self, game: chess.Board()=None, to_play=1, turn=0):
        """Initialize the chess board """
        if game is None:
            self.Board = chess.Board()
        else:
            self.Board = game
        self.turn = turn
        self.to_play = to_play

    def is_finished(self) -> bool:
        """Check if the game is finished (e.g. checkmate, stalemate, draw)
        Return True if the game is finished, False otherwise"""
        return (self.Board.is_variant_draw or self.Board.is_variant_loss or self.Board.is_variant_win)

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

    def step(self, action) -> None:
        """Takes in a move and performs it on the board"""
        self.Board.push(action)

    def step_back(self):
        """Takes a step back in the game"""
        self.Board.pop()

    def get_state(self):
        """Gives the current state of the board, used to visualize board"""
        return self.Board
    
    def render(self):
        """Renders the board"""
        print(self.Board)

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