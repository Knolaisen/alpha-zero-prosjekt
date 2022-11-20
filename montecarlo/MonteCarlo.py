from math import log,sqrt
import math
import random
from select import select
from Node import Node
from collections import defaultdict


class MonteCarlo():
    
    def __init__(self, depth) -> None:
        self.depth = depth
        self.stateDict = defaultdict(list)
        
    def UCB1(self, node):
        if(node.N == 0):
            return sqrt(2)
        if(node.P == None):
            return node.v / node.N
        return node.v / node.N + sqrt(2) * sqrt(log(node.P.N)/node.N) 
        
    def Search(self, state):
        Currdepth = 0
        node = Node(state)
        if node not in self.stateDict:
            self.stateDict[node] = []
            node.make_children(self.stateDict)

        while Currdepth <= self.depth:
            leaf = self.Select(node)
            child = self.Expand(leaf, Currdepth)
            result = self.Simulate(child)
            self.Back_propagate(result, child)
            Currdepth += 1
        
        value, move = -math.inf, None
        top_moves = []
        for i in self.stateDict[node]:
            if i.v > value:
                top_moves.clear()
                value = i.v
                top_moves.append(i.get_move())
            elif i.v == value:
                top_moves.append(i.get_move())
        
        move = random.choice(top_moves)
        return move
    
    #node.v = total utility of node
    #node.N = number of playouts of node
    #node.P = node parent
    
    def Select(self, node):
        max_ucb = -math.inf
        select_child = None
        best_children = []
        for i in self.stateDict[node]:
            current_ucb = self.UCB1(i)
            if current_ucb > max_ucb:
                best_children.clear()
                max_ucb = current_ucb
                best_children.append(i)
            elif current_ucb == max_ucb:
                best_children.append(i)
        if best_children:
            select_child = random.choice(best_children)
        else:
            select_child = random.choice(self.stateDict[node])
        return select_child
    
    def Expand(self, node,depth):
        node.make_children(self.stateDict)
        if len(self.stateDict[node])==0 or depth>self.depth:
            return node
        max_ucb = -math.inf
        select_child = None
        for i in self.stateDict[node]:
            curr_ucb = self.UCB1(i)
            if curr_ucb > max_ucb:
                max_ucb = curr_ucb
                select_child = i
        return self.Expand(select_child, depth+1)
    
    def Simulate(self, node):
        if(node not in self.stateDict):
            return
        if(node.state.is_game_over()):
            if(node.state.result()=='1-0'):
                return (1)
            elif(node.state.result() == '0-1'):
                return(-1)
            return 0.5
        #print(node.children)
        try:
            random_child = random.choice(self.stateDict[node])
        except Exception:
            print(self.stateDict[node])
        return self.Simulate(random_child)

    
    def Back_propagate(self,result, node):
        while(node.P != None):
            node.v += result
           # node.N += 1
            node = node.P
