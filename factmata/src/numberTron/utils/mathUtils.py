'''
Created on 11 Ara 2016

@author: YEB
'''
@staticmethod
def within(self, arg, trueval, margin):
        upper = trueval + margin * trueval
        lower = trueval - margin * trueval
        return (arg >= lower and arg <= upper)