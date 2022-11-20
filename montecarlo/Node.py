import copy
import chess

class Node():
    
    def __init__(self, board, parent=None) -> None:
        self.state = board
        self.P = parent
        self.N = 0
        self.n = 0
        self.v = 0
    
    def make_children(self, stateDict):
        for i in self.state.legal_moves:
            board = copy.copy(self.state)
            board.push(i)
            if self not in stateDict:
                stateDict[self] = []
            if hash(board.fen()[:-4]) in stateDict:
                stateDict[self] = stateDict[hash(board.fen()[:-4])]
                return
            else:    
                stateDict[self].append(Node(board))
            
    
    def __hash__(self) -> int:
        return hash(self.state.fen()[:-4])
    
    def __eq__(self, other) -> bool:
        return self.__hash__() == other.__hash__()
    
    def get_move(self):
        return self.state.peek()