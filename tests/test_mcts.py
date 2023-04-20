import pytest
from chess_handler import ChessStateHandler
from mcts import backpropagation, expansion, selection, simulation, ucb, generate_test_data
from node import Node
from mock_state_handler import mock_state_handler


def test_selection():
    # create parent node
    parent_node = Node(state=None, parent=None)
    # create child nodes with different rewards
    child1 = Node(state=None, parent=parent_node)
    child1.set_wins(2)
    child1.set_visits(3)
    child2 = Node(state=None, parent=parent_node)
    child2.set_wins(4)
    child2.set_visits(5)
    child3 = Node(state=None, parent=parent_node)
    child3.set_wins(1)
    child3.set_visits(2)
    # perform selection
    best_child = selection(parent_node)
    # assert that the best child is child2 with the highest UCB score
    assert best_child == child2

def test_ucb():
    node = Node(state=None, parent=Node(state=None, parent=None))
    node.set_wins(1)
    node.set_visits(1)
    node.get_parent().set_visits(2)
    assert ucb(node) == pytest.approx(2.44948974278, 0.001)


def test_expansion(mock_state_handler):
    node = Node(state=None, parent=None)
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
    root = Node(state=state, parent=None)
    
    # Ensure that the simulation function returns the correct winner for a finished game state
    assert simulation(root) == 1

def test_backpropagation(mock_state_handler):
    node = Node(state=None, parent=None)
    result = 1
    backpropagation(node, result)
    assert node.get_visits() == 1
    assert node.get_wins() == 1
    
    result = -1
    backpropagation(node, result)
    assert node.get_visits() == 2
    assert node.get_wins() == 0

def test_mcts(mock_state_handler):
    # create root node
    root = Node(state=None, parent=None)
    # perform MCTS

    # assert that the root node has children
    assert root.has_children()

def test_generate_test_data():
    print("Test generate test data:")

    chessHandler = ChessStateHandler()
    # create root node
    root = Node(chessHandler)
    
    generate_test_data(root, 20, 50)

if __name__ == "__main__":
    # test_selection()
    test_generate_test_data()