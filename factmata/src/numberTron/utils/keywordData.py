'''
Created on 10.12.16

@author: Yunus Emrah Bulut

'''
import json

class KeywordData:
    
    keyWordFile = "../../../data/keywords.json"
    modifiers = [ "change", "up", "down", "males", "females", "male",
            "female", "growth", "increase", "decrease", "decreased", "increased", "changed", "grown", "grew", "surge",
            "surged", "rose", "risen"]
    relName = None
    KEYWORDS = []
    KEYWORD_SET = None
    NUM_RELATIONS = None
    REL_KEYWORD_MAP = {}
    keywordJson = None
    KEYWORD_SET = set()
    
    with open(keyWordFile, 'r') as f:
        keywordJson = json.load(f)
        KEYWORDS = []

        NUM_RELATIONS = len(keywordJson.keys())

        relName = keywordJson.keys()

        for rel in relName:
            value = keywordJson[rel]
            REL_KEYWORD_MAP[rel] = value
            KEYWORDS.append(value)
            for kw in value:
                KEYWORD_SET.add(kw)
      
