import pytest
from monteCarloTreeSearch.MonteCarloTreeSearch import backpropagation, expansion, selection, simulation, ucb
from monteCarloTreeSearch.TreeNode import TreeNode
from tests.monteCarloTreeSearch.MockStateHandler import mock_state_handler


def test_selection():
    # create parent node
    parent_node = TreeNode(state=None, parent=None)
    # create child nodes with different rewards
    child1 = TreeNode(state=None, parent=parent_node)
    child1.set_wins(2)
    child1.set_visits(3)
    child2 = TreeNode(state=None, parent=parent_node)
    child2.set_wins(4)
    child2.set_visits(5)
    child3 = TreeNode(state=None, parent=parent_node)
    child3.set_wins(1)
    child3.set_visits(2)
    # perform selection
    best_child = selection(parent_node)
    # assert that the best child is child2 with the highest UCB score
    assert best_child == child2

def test_ucb():
    node = TreeNode(state=None, parent=TreeNode(state=None, parent=None))
    node.set_wins(1)
    node.set_visits(1)
    node.get_parent().set_visits(2)
    assert ucb(node) == pytest.approx(2.44948974278, 0.001)


def test_expansion(mock_state_handler):
    node = TreeNode(state=None, parent=None)
    assert len(node.get_children()) == 0
    expansion(node, mock_state_handler)
    assert len(node.get_children()) == 1
    assert node.get_children()[0].get_parent() == node
    assert node.get_children()[0].get_state() == None

def test_simulation():
    # Set up a simple tree node with a finished game state
    state = mock_state_handler()
    state.is_finished.return_value = True
    state.get_winner.return_value = 1
    root = TreeNode(state=state, parent=None)
    
    # Ensure that the simulation function returns the correct winner for a finished game state
    assert simulation(root) == 1

def test_backpropagation(mock_state_handler):
    node = TreeNode(state=None, parent=None)
    result = 1
    backpropagation(node, result)
    assert node.get_visits() == 1
    assert node.get_wins() == 1
    
    result = -1
    backpropagation(node, result)
    assert node.get_visits() == 2
    assert node.get_wins() == 0

    
if __name__ == "__main__":
    test_selection()