import numpy as np


class ConnectFour():
    def __init__(self, shape: tuple = (6, 7), counter=0):
        self.shape = shape
        self.board = np.zeros(shape)
        self.counter = counter

    def get_board(self) -> np.array:
        return self.board

    def _get_next_open_row(self, column: int) -> int:
        # Check for free square from bottom-up
        for row_idx in range(self.shape[0]-1, -1, -1):

            # Return the row immediately if free
            if self.board[row_idx][column] == 0:
                return row_idx

        # Return None if no free squares in given column
        return None

    def is_legal_move(self, column: int) -> bool:
        if column < 0 or column >= self.shape[1]:
            raise Exception(
                f"Column {column} is greater than board with shape {self.shape}")

        if self._get_next_open_row(column) is None:
            return False
        else:
            return True

    def move(self, column: int) -> None:
        if column is None:
            raise Exception(
                "Missing required arguments (requires column)")

        # Validate player
        #if not (player == -1 or player == 1):
        #    raise Exception("Player must be -1 or 1")

        # Check if legal move
        row = self._get_next_open_row(column)
        if row is None:
            raise Exception(f"Illegal move. Column {column} is full")

        # Do move
        self.board[row][column] = 1
        self.counter += 1
    
    def switch_sides(self):
        self.board[(self.board > 0) | (self.board < 0)] *= -1

    def get_available_moves(self) -> list[int]:
        return [column for column in range(self.shape[1]) if self._get_next_open_row(column) is not None]

    def get_illegal_moves(self) -> list[int]:
        return [column for column in range(self.shape[1]) if self._get_next_open_row(column) is None]

    def is_win_state(self):
        # Test rows
        for i in range(self.shape[0]):
            for j in range(self.shape[1] - 3):
                value = sum(self.board[i][j:j + 4])
                if abs(value) == 4:
                    winner = 1 if value > 0 else -1
                    return (True, winner)

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*self.board)]
        for i in range(self.shape[1]):
            for j in range(self.shape[0] - 3):
                value = sum(reversed_board[i][j:j + 4])
                if abs(value) == 4:
                    winner = 1 if value > 0 else -1
                    return (True, winner)

        # Test diagonal
        for i in range(self.shape[0] - 3):
            for j in range(self.shape[1] - 3):
                value = 0
                for k in range(4):
                    value += self.board[i + k][j + k]
                    if abs(value) == 4:
                        winner = 1 if value > 0 else -1
                        return (True, winner)

        reversed_board = np.fliplr(self.board)
        # Test reverse diagonal
        for i in range(self.shape[0] - 3):
            for j in range(self.shape[1] - 3):
                value = 0
                for k in range(4):
                    value += reversed_board[i + k][j + k]
                    if abs(value) == 4:
                        winner = 1 if value > 0 else -1
                        return (True, winner)

        return (False, 0)

    def is_game_over(self) -> bool:
        return len(self.get_available_moves()) == 0 or self.is_win_state()[0]

    def trim_and_normalize(self, array):
        
        illegal_moves = self.get_illegal_moves()
        legal_moves = self.get_available_moves()

        # Replace small values with zero to stabilize norm function.  
        # This is an attempt to fix. "RuntimeWarning: divide by zero encountered in divide" and "RuntimeWarning: invalid value encountered in divide"
        array[array < 1e-5] = 0 

        if illegal_moves:
            for i in illegal_moves: # Replacec illegal moves with value 0
                array[i] = 0
            if (array == 0).all(): # Enter if array is completely empty and cannot be normalized.
                # Fills empty array with available moves to prevent "RuntimeWarning: divide by zero encountered in divide" from killing the neural network.
                for i in legal_moves:
                    array[i] = 1
            array = array/np.linalg.norm(array)
            return array # Return normalized array (NB!: does not sum to 1 it seems)
            
        return array
