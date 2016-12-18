'''
Created on 17.12.16

@author: Yunus Emrah Bulut
'''
from dependencyparser import DependencyParser
from namedentityrecognizer import NamedEntityRecognizer

class NumberRuleAlgorithm:
    
    def __init__(self):
        self.modtypesPath = "../resources/modtypes"
    
    def parse(self, sentence):
        dp = DependencyParser()
        return dp.parse(sentence)
    
    def ner(self, sentence):
        ner = NamedEntityRecognizer()
        return ner.parse(sentence)
    
    def getDependencyGraph(self, parse):
        dp = DependencyParser()
        return dp.buildGraph(parse)
        
    def getCandidateEntities(self, parse, countryList):
        dp = NamedEntityRecognizer()
        return dp.findCandidateEntities(parse, countryList)
    
    def getNumbers(self, gr):
        return gr.getNumberNodes()
    
    def checkIfKeywordExists(self, keyword, gr, entityPos, numberPos):
        sp = gr.getShortestPathBetween(entityPos, numberPos)
        if len(sp)==2:
            return True
        for n in sp:
            if (n == numberPos) or (n==entityPos):
                continue
            if gr.nodes[n].word == keyword:
                return True
        sp2 = gr.getShortestPath()[0]
        with open(self.modtypesPath, 'r') as f:
            modtypes = f.read().splitlines()
            for n in sp:
                if n != entityPos and n != numberPos:
                    neighbours = sp2[n,:]
                    for i in range(len(neighbours)):
                        if neighbours[i] == 1 and gr.nodes[i].rel in modtypes and gr.nodes[i].word == keyword:
                            return True
                
        return False
    
    def run(self, keyword, sentence):
        
        parse = self.parse(sentence)
        ner = self.ner(sentence)
        gr = self.getDependencyGraph(parse)
        
        countries = []
        with open("../resources/countries_list",'r') as f:
            for country in f.read().splitlines():
                countries.append(country)
                    
        candidates = self.getCandidateEntities(ner, countries)
        numbers = self.getNumbers(gr)
        relations =   []
        for cand in candidates:
            pos = gr.getPosition(cand)
            for p in pos:
                for num in numbers:
                    if self.checkIfKeywordExists(keyword, gr, p, num):
                        relations.append([keyword, gr.nodes[p].word, gr.nodes[num].word])
        return relations