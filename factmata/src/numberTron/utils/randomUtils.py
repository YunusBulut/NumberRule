'''
Created on 11.12.16

@author: Yunus Emrah Bulut
'''
import random

class RandomUtils:
    
    @staticmethod
    def coinToss(self, bias):
        
        toss = random.uniform(0, 1)
        return toss <= bias
