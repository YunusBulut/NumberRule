'''
Created on 10.12.16
@author: Yunus Emrah Bulut

This module mimics the LRGraph and Number classes of NEO-IE/numbertron.
@link: https://github.com/NEO-IE/numbertron/blob/master/src/main/java/iitb/neo/training/ds/LRGraph.java

'''
class Number:

    def __init__(self, zlist, num = -1.0):
        self.num = float(num)
        self.zlist = zlist

class SparseBinaryVector:

    def __init__(self, ids = [], num = 0): 
        self.ids = ids
        self.num = num
      
    def reset(self):
        SparseBinaryVector.num = 0 #shouldn't we also reset the array to an empty list?
    
    def copy(self):
        sbv = list(SparseBinaryVector.ids)
        return sbv
    
    def dotProduct(self, sbv):
        return sum(i[0] * i[1] for i in zip(SparseBinaryVector.ids, sbv))
      
class Graph:
    
    MNT_CAPACITY = 1
    NUM_RELS = 11
    
    def __init__(self):
        self.mentionIDs = [None]*self.MNT_CAPACITY
        self.Z = [None]*self.MNT_CAPACITY
        self.features = SparseBinaryVector(num = self.MNT_CAPACITY)
        self.numFeatures = SparseBinaryVector(num = self.MNT_CAPACITY)
        self.numMentionIDs = [None]*self.MNT_CAPACITY
        self.N = [None]*self.MNT_CAPACITY
        self.numMentions = 0
        self.random = 0
        self.numNodesCount = 0
    
    def clear(self):
        Graph.numMentions = 0;
        Graph.numNodesCount = 0;

    def setCapacity(self, targetSize):
            newMentionIDs = [None]*targetSize
            newZ = [None]*targetSize
            newFeatures = SparseBinaryVector(num = targetSize)
            if Graph.numMentions > 0:
                Graph.mentionIDs = list(newMentionIDs)
                Graph.Z = list(newZ)
                Graph.features = list(newFeatures)
          
            Graph.mentionIDs = newMentionIDs
            Graph.Z = newZ
            Graph.features = newFeatures

    def setNumCapacity(self, targetSize):
            newNumMentionIDs = [None]*targetSize
            newN = [None]*targetSize
            newNumFeatures = SparseBinaryVector(num = targetSize)
            if Graph.numNodesCount > 0:
                Graph.numMentionIDs = list(newNumMentionIDs)
                Graph.N = list(newN)
                Graph.numFeatures = list(newNumFeatures)
          
            Graph.numMentionIDs = newNumMentionIDs;
            Graph.N = newN;
            Graph.numFeatures = newNumFeatures;
