

class TreeNode():
    """
    A class for the nodes, which among other things contains the board state
    """
    def __init__(self, state, parent: "TreeNode") -> None:
        """
        Main condstructor, takes in state and parent node.
        setting all num values as zero. 
        """
        # _ makes the variables private
        self._state = state
        self._isTerminal = state.isTerminal()
        self._isFullyExpanded = self.isTerminal
        self._parent = parent
        self._numVisits = 0
        self._numWins = 0
        self._numDraws = 0
        self._numLoses = 0
        self._children = []

    def get_numVisits(self) -> int:
        """
        Returns numVisits
        """
        return self.num_visists

    def get_numWins(self) -> int:
        """
        Returns numWins
        """
        return self.num_wins

    def get_numDraws(self) -> int:
        """
        Returns numDraws
        """
        return self.num_draws

    def get_numLoses(self) -> int:
        """
        Returns numLoses
        """
        return self._numLoses

    def get_parent(self) -> "TreeNode":
        """
        Returns parent
        """
        return self.parent
    
    def is_leaf(self) -> bool:
        """
        Returns True if self is leaf (has no children)
        """
        return bool(len(self.children))
    
    def is_parent(self) -> bool:
        """
        Returns True if self is parent (has children)
        """
        return bool(self.parent)
    
    def get_children(self) -> list:
        """
        Returns list of children
        """
        return self._children
    
    def has_children(self) -> bool:
        """
        Returns True if has children
        """
        return bool(len(self._children))