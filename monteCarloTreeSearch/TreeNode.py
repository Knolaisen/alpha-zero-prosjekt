from state import state_handler

class TreeNode():
    """
    A class for the nodes, which among other things contains the board state
    """
    def __init__(self, state, parent=None) -> None:
        """
        Main constructor, takes in state and parent node.
        setting all num values as zero. 
        """
        # _ makes the variables private
        self._state = state
        self._parent : "TreeNode" = parent
        self._visits = 0
        self._wins = 0
        self._draws = 0
        self._loses = 0
        self._children = []
        if parent is not None:
            parent.add_child(self)

    def get_visits(self) -> int:
        """
        Returns numVisits
        """
        return self._visits

    def get_wins(self) -> int:
        """
        Returns numWins
        """
        return self._wins

    def get_draws(self) -> int:
        """
        Returns numDraws
        """
        return self._draws

    def get_loses(self) -> int:
        """
        Returns numLoses
        """
        return self._loses

    def get_parent(self) -> "TreeNode":
        """
        Returns parent
        """
        return self._parent
    
    def is_leaf(self) -> bool:
        """
        Returns True if self is leaf (has no children)
        """
        return bool(len(self._children))
    
    def is_parent(self) -> bool:
        """
        Returns True if self is parent (has children)
        """
        return bool(self._parent)
    
    def get_children(self) -> list["TreeNode"]:
        """
        Returns list of children
        """
        return self._children
    
    def has_children(self) -> bool:
        """
        Returns True if has children
        """
        return bool(len(self._children))
    
    def add_child(self, child:"TreeNode") -> None:
        """
        Add a node to children
        """
        self._children.append(child)
        child._parent = self

    def get_state(self) -> state_handler:
        """
        Returns the state of the node
        """
        return self._state
    
    def set_fully_expanded(self, value:bool) -> None:
        """
        Set the value of fully expanded
        """
        self._is_fully_expanded = value
        
    def set_wins(self, wins) -> None:
        '''
        Set the value of wins
        '''
        self._wins = wins

    def set_visits(self, visits) -> None:
        '''
        Set the value of visits
        '''
        self._visits = visits
    
    def has_parent(self) -> bool:
        return self._parent != None
    
    def add_visits(self) -> None:
        self._visits = self._visits + 1

    def add_win(self) -> None:
        self._wins = self._wins + 1
        