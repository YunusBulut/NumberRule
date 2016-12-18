'''
Created on 15.12.16

@author: Yunus Emrah Bulut
'''
import gzip
import os
class BufferedIOUtils:
    
    @staticmethod
    def getBufferedReader(inputFileName):
        
        filename, file_extension = os.path.splitext(inputFileName)
        if file_extension == ".gz":
            return gzip.open(inputFileName,'rb')
        else:
            return open(inputFileName,'r')
    
    @staticmethod
    def getBufferedWriter(outputFileName):
        filename, file_extension = os.path.splitext(outputFileName)
        if file_extension == ".gz":
            return gzip.open(outputFileName,'wb')
        else:
            return open(outputFileName,'r')
