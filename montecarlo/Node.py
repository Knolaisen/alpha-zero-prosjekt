import copy
import chess

class Node():
    
    def __init__(self, stateDict, board, depth=0, parent=None) -> None:
        self.depth = depth
        self.depth_limit = 2
        self.state = board
        self.children = self.make_children(depth, stateDict)
        self.P = parent
        self.N = 0
        self.n = 0
        self.v = 0
    
    def make_children(self, depth, stateDict):
        if self.depth > self.depth_limit:
            return []
        children = []
        for i in self.state.legal_moves:
            board = copy.copy(self.state)
            board.push(i)
            if hash(board.fen()[:-4]) in stateDict:
                children.append(stateDict[hash(board.fen()[:-4])])
            else:    
                children.append(Node(stateDict, board, depth+1, self))
            
        return children
    
    def __hash__(self) -> int:
        return hash(self.state.fen()[:-4])
    
    def __eq__(self, other) -> bool:
        return self.__hash__() == other.__hash__()
    
    def get_move(self):
        return self.state.peek()