## Dependencies
import math
import numpy as np
from TreeNode import TreeNode
from state import state_handler
import random


def monte_carlo_tree_search():
    pass

def selection(node: TreeNode):
    child_nodes = node.get_children
    best_child = None
    best_node_value = 0

    while child_nodes:
        for child_node in child_nodes:
            if (best_node_value < ucb(child_node)):
                best_child = child_node
                best_node_value = ucb(child_node)
        return selection(best_child)
    return node

def expansion(node:TreeNode):
    """
    In this process, a new child node is added to the tree to that
    node which was optimally reached during the selection process.
    For now only makes random moves to propegate further
    """
    legal_actions = state_handler.get_legal_actions()
    rnd = random.randint(0, len(legal_actions)-1) #Choose random action

    #Checking if any of the children for optimal node is same as action
    for action in legal_actions:
        for child in node.get_children():
            if child.get_state() != action:
                new_node = TreeNode(node.get_state().move(action))
                node.add_child(new_node)



    pass

def simulation():
    """
    In this process, a simulation is performed by choosing moves or strategies until a result or predefined state is achieved.
    """
    pass

def backpropergation():
    """
    After determining the value of the newly added node, the remaining tree must be updated. So, the backpropagation process is performed, where it backpropagates from the new node to the root node. During the process, the number of simulation stored in each node is incremented. Also, if the new node’s simulation results in a win, then the number of wins is also incremented.
    """
    pass

def upper_condidence_bound(empiricalMean: float, visitationOfParentNode: int, visitationOfChildNode: int ) -> float:
    constant = math.sqrt(2)
    exploredness = math.sqrt(math.log(visitationOfParentNode)/visitationOfChildNode)
    return empiricalMean + constant * exploredness
 
def ucb(node: TreeNode):
    """
    Takes in node and returns upper confidence bound based on parent node visits and node visits
    """
    exploration_parameter = math.sqrt(2)
    exploitation = node.get_numWins()/node.get_numVisits()
    exploration = np.sqrt(np.log(node.get_parent().get_numVisits())/node.get_numVisits())
    return exploitation + exploration_parameter*exploration


if __name__ == "__main__":
    def selection_test():
        root_node = TreeNode("1")
        root_node._wins = 11
        root_node._visits = 21 

        
        wanted_node1 = TreeNode("2", root_node)
        wanted_node1._wins = 7
        wanted_node1._visits = 10 

        wanted_node2 = TreeNode("3", root_node)
        wanted_node2._wins = 3
        wanted_node2._visits = 8 


        wanted_node4 = TreeNode("4", wanted_node1)
        wanted_node4._wins = 2
        wanted_node4._visits = 4 

        wanted_node5 = TreeNode("5", wanted_node1)
        wanted_node5._wins = 1
        wanted_node5._visits = 6


        wanted_node6 = TreeNode("6", wanted_node5)
        wanted_node6._wins = 2
        wanted_node6._visits = 3


        wanted_node7 = TreeNode("7", wanted_node5)
        wanted_node7._wins = 3
        wanted_node7._visits = 3

        selcted_node = selection(root_node)
        print(selcted_node.get_state)



        



        
