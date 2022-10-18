import copy
import chess

class Node():
    
    def __init__(self, board, depth=0, parent=None) -> None:
        self.depth = depth
        self.depth_limit = 2
        self.state = board
        self.children = self.make_children(depth)
        self.P = parent
        self.N = 0
        self.n = 0
        self.v = 0
    
    def make_children(self,depth):
        if self.depth > self.depth_limit:
            return []
        children = []
        for i in self.state.legal_moves:
            board = copy.copy(self.state)
            board.push(i)
            children.append(Node(board, depth+1, self))
            
        return children
    
    def __hash__(self) -> int:
        return hash(self.state)

    
    def get_move(self):
        return self.state.peek()