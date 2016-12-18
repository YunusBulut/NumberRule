'''
Created on 17.12.16

@author: Yunus Emrah Bulut
'''
import numpy as np
from scipy.sparse import csgraph
import math
class Graph:
    
    def __init__(self, numberOfNodes):
        self.numberOfNodes = numberOfNodes
        self.nodes = [None]*numberOfNodes
        self.sparseMatrix = np.zeros((self.numberOfNodes,self.numberOfNodes))
        self.modtypes = "../resources/modtypes"
    
    def getNodes(self):
        return self.nodes
    
    def getSparseMatrix(self):
        return self.sparseMatrix
    
    def getPosition(self, word):
        positions = []
        for node in self.nodes:
            if node.word == word:
                positions.append(node.pos)
        return positions
    
    def insertNode(self, node):
        if node.pos !=0:
            self.nodes[node.pos] = node
            for v in list(node.deps.values()):
                self.sparseMatrix[v,node.pos] = 1
                self.sparseMatrix[node.pos,v] = 1
        else:
            self.nodes[node.pos] = node
            for v in list(node.deps.values()):
                self.sparseMatrix[v,node.pos] = 1
                self.sparseMatrix[node.pos,v] = 1
    
    def getShortestPath(self):
        return csgraph.shortest_path(self.getSparseMatrix(), return_predecessors = True, directed = False)
    
    def getShortestPathBetween(self, i, j):
        sp = []
        spm = self.getShortestPath()[1]
        step = spm[i,j]
        if step == math.inf or step<0:
            return sp
        sp.append(j)
        while True:
            sp.append(step)
            if step==i:
                break
            step = spm[i,step]
        return sp
    
    def getNumberNodes(self):
        
        '''
        with open(self.modtypes,'r') as f:
            mods = f.read().splitlines()
        '''
        num = []
        
        mods = ['nummod', 'compound']
        for n in self.nodes:
            if n.rel in mods:
                num.append(n.pos)
        
        return num
    
    class Node:
        def __init__(self, deps, pos, tag, word, rel):
            self.deps = deps
            self.pos = pos
            self.tag = tag
            self.word = word
            self.rel = rel
            
    
        
