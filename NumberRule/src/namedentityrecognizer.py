'''
Created on 17.12.16

@author: Yunus Emrah Bulut
'''
from nltk.tag.stanford import StanfordNERTagger

class NamedEntityRecognizer:
    
    def __init__(self):
        self.path_to_jar = 'C:/Users/YEB/Desktop/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz'
        self.path_to_models_jar = 'C:/Users/YEB/Desktop/stanford-ner-2015-12-09/stanford-ner.jar'
        self.ner = StanfordNERTagger(self.path_to_jar, self.path_to_models_jar)
    
    def parse(self, sentence):
        return self.ner.tag(sentence.split(" "))
        
    def findCandidateEntities(self, parse, entities):    
        matchedEntities = []
        for entity in entities:
            for tpl in parse:
                if entity == tpl[0]:
                    matchedEntities.append(entity)
        
        return matchedEntities 
        
if __name__=='__main__':
    dp = NamedEntityRecognizer()
    countries = []
    with open("../resources/countries_list",'r') as f:
        for country in f.read().splitlines():
            countries.append(country)
    print(dp.parse("The GDP of Australia is about 400 billion"))    
    print(dp.findCandidateEntities(dp.parse("The GDP of Australia is about 400 billion"), countries))