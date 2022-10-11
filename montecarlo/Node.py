import copy
import chess

class Node():
    
    def __init__(self, board:chess.Board()) -> None:
        self.state = board
        self.children = self.make_children(self.state.legal_moves)
        self.P = self.get_patent()
        self.N = 0
        self.n = 0
        self.V = 0
    
    def make_children(self,legal_moves):
        board = copy(self.state)
        children = []
        for i in legal_moves:
            children.append(Node(board.push(i)))
        return children
    
    def __hash__(self) -> int:
        return hash(self.state.legal_moves)
    
    def get_patent(self):
        parent = copy(self.state)
        parent.pop()
        return parent
    
    def get_move(self):
        return self.state.peek()