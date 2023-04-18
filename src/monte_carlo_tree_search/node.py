from state_handler.state import StateHandler


class Node():
    """
    A class for the nodes, which among other things contains the board state
    """

    def __init__(self, state: StateHandler, parent:"Node"=None) -> None:
        """
        Main constructor, takes in state and parent node.
        setting all num values as zero. 
        """
        # _ makes the variables private
        self._state: StateHandler = state
        self._parent: "Node" = parent
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

    def get_parent(self) -> "Node":
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

    def get_children(self) -> list["Node"]:
        """
        Returns list of children
        """
        return self._children

    def has_children(self) -> bool:
        """
        Returns True if has children
        """
        return bool(len(self._children))

    def add_child(self, child: "Node") -> None:
        """
        Add a node to children
        """
        self._children.append(child)
        child._parent = self

    def get_state(self) -> StateHandler:
        """
        Returns the state of the node
        """
        return self._state

    def set_fully_expanded(self, value: bool) -> None:
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

    def add_draw(self) -> None:
        self._wins += 0.5

    def add_reward(self, reward: int) -> None:
        """
        Adds a reward to the node
        """
        if reward == 1:
            self.add_win()
        elif reward == 0:
            self.add_draw()
        elif reward == -1:
            pass
        else:
            raise ValueError("Reward must be 1, 0 or -1")
