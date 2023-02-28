## Dependencies
import math
import numpy as np
import TreeNode


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

def expansion(node):
    """
    In this process, a new child node is added to the tree to that node which was optimally reached during the selection process.
    """
    #if not node.isTerminal():
        #Something like for i in available moves:
            #nc = treeNode(newState, self)
            #children.append(nc)
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
        
        unwanted_node1 = TreeNode("2", root_node)
        unwanted_node2 = TreeNode("3", root_node)
        
        wanted_node_parent1 = TreeNode("4", unwanted_node1)
        wanted_node_parent2 = TreeNode("5", unwanted_node1)


        



        
