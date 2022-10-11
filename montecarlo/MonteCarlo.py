import chess
from cmath import sqrt
from math import log2
import math
from random import random
import Node


class MonteCarlo():
    
    def __init__(self, depth) -> None:
        self.UCB1 = lambda node: node.v/node.N + sqrt(2) * sqrt(log2(node.P.N)/node.N) 
        self.depth = depth
        
    def Search(self, state:chess.Board):
        Currdepth = 0
        node = Node(state)
        while Currdepth <= self.depth:
            leaf = self.Select(node)
            child = self.Expand(leaf)
            result = self.Simulate(child)
            self.Back_propagate(result, child)
            Currdepth += 1
        return 
    
    #node.v = total utility of node
    #node.N = number of playouts of node
    #node.P = node parent
    
    def Select(self, node):
        max_ucb = -math.inf
        select_child = None
        for i in node.children:
            current_ucb = self.UCB1(1)
            if current_ucb > max_ucb:
                max_ucb = current_ucb
                select_child = i
        return select_child
    
    def Expand(self, node,depth):
        if not node.children or depth>self.depth:
            return node
        max_ucb = -math.inf
        select_child = None
        for i in node.children:
            curr_ucb = self.UCB1(i)
            if curr_ucb > max_ucb:
                max_ucb = curr_ucb
                select_child = i
        return self.Expand(select_child)
    
    def Simulate(self, node):
        if(node.state.is_game_over()):
            if(node.state.result()=='1-0'):
                return (1)
            elif(node.state.result() == '0-1'):
                return(0)
            return 0.5
        random_child = random.choice(node.children)
        return self.Simulate(random_child)
    
    def Back_propagate(self,result, node):
        while(node.P != None):
            node.V += result
            node = node.P
