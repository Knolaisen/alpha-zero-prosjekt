import numpy as np


class ConnectFour():
    def __init__(self, rows:int = 6, columns:int = 7):
        self.rows = rows
        self.columns = 7
        self.board = np.zeros((rows, columns))

    def get_reward(self):
        pass

    def get_board(self):
        print(self.board)
    
    def get_available_moves(self):
        pass
    
    def move(self):
        pass
    
    def is_win_state(self):
        pass
    
    