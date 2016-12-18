'''
Created on 17.12.16

@author: Yunus Emrah Bulut
'''
from nltk.parse.stanford import StanfordNeuralDependencyParser
import graph

class DependencyParser:
    
    def __init__(self):
        self.path_to_jar = 'C:/Users/YEB/Desktop/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar'
        self.path_to_models_jar = 'C:/Users/YEB/Desktop/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0-models.jar'
        self.dependency_parser = StanfordNeuralDependencyParser(path_to_jar=self.path_to_jar, path_to_models_jar=self.path_to_models_jar)
        self.dependency_parser.java_options = '-Xmx3024m'
   
    def parse(self, sentence):
        return list(self.dependency_parser.raw_parse(sentence))[0]

    def buildGraph(self, parse):
        
        numOfNodes = len(parse.nodes)
        gr = graph.Graph(numOfNodes)
        for i in range(numOfNodes):
            if parse.nodes[i]['tag'] !='TOP':
                gr.insertNode(gr.Node(parse.nodes[i]['deps'], parse.nodes[i]['address'], parse.nodes[i]['tag'], parse.nodes[i]['word'], parse.nodes[i]['rel']))
            else:
                gr.insertNode(gr.Node(parse.nodes[i]['deps'], parse.nodes[i]['address'], parse.nodes[i]['tag'], None, None))
        return gr    
                

if __name__=='__main__':
    dp = DependencyParser()
    print(dp.parse("The GDP of Australia is about 400 billion"))
    print(dp.buildGraph(dp.parse("The GDP of Australia is about 400 billion")).getShortestPathBetween(4,7))