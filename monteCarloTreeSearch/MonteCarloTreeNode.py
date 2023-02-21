

class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.numWins = 0
        self.numDraws = 0
        self.numLoses = 0
        self.children = {}

    def is_leaf(self):
        return self.children.empty()
    
    def parent(self):
        return self.parent  