## Dependencies
import math
import numpy as np
import TreeNode


def monte_carlo_tree_search():
    pass

def selection():
    pass

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
 
def ucb(node: TreeNode ):
    
    pass
