'''
Created on 11.12.16

@author: Yunus Emrah Bulut
'''
import numberTron.algorithms as alg

class RelationMetadata:
    
    unitMap = {}
    unitMap["NA"] = "xxxxxx" #hopefully will never be found
    unitMap["AGL"] = "square metre"
    #unitMap["N_AGL"] = "square metre"
    unitMap["FDI"] = "united states dollar"
    unitMap["GOODS"] = "united states dollar"
    unitMap["GDP"] = "united states dollar"
    #unitMap["N_FDI_GOODS_GDP"] = "united states dollar"
    unitMap["ELEC"] = "joule"
    #unitMap["N_ELEC"] = "joule"
    unitMap["CO2"] = "kilogram"
    #unitMap["N_CO2"] = "kilogram"
    #unitMap["DIESEL"] = "united states dollar per litre"
    unitMap["INF"] = "percent"
    unitMap["INTERNET"] = "percent"
    #unitMap["N_INF_INTERNET"] = "percent"
    unitMap["LIFE"] = "second"
    #unitMap["N_LIFE"] = "second"
    unitMap["POP"] = ""
    #unitMap["N_POP"] = ""
    
    relMap = {}
    relMap["AG.LND.TOTL.K2"] = "AGL"
    relMap["BN.KLT.DINV.CD"] = "FDI"
    relMap["BX.GSR.MRCH.CD"] = "GOODS"
    relMap["EG.ELC.PROD.KH"] = "ELEC"
    relMap["EN.ATM.CO2E.KT"] = "CO2"
    #relMap["EP.PMP.DESL.CD"] = "DIESEL"
    relMap["FP.CPI.TOTL.ZG"] = "INF"
    relMap["IT.NET.USER.P2"] = "INTERNET"
    relMap["NY.GDP.MKTP.CD"] = "GDP"
    relMap["SP.DYN.LE00.IN"] = "LIFE"
    relMap["SP.POP.TOTL"] = "POP"
    
    @staticmethod
    def getUnit(rel):
            return RelationMetadata.unitMap[rel]
    
    @staticmethod
    def getRelations():
        return RelationMetadata.unitMap.keys()
    
    @staticmethod
    def getShortenedRelation(rel):
        return RelationMetadata.relMap[rel]
    
    @staticmethod
    def getWorldBankRels():
        return RelationMetadata.relMap.keys()
    

class GoldDB:
    
    # @Todo: Read the gold database path from the json file.
    
    goldDBFileLoc = None
    goldDB = None
    countries = None
    K = None
    MARGIN = None
    
    @staticmethod
    def initializeGoldDB(goldDBName, topK, matchMargin):
        GoldDB.K = topK
        GoldDB.MARGIN = matchMargin
        GoldDB.goldDBFileLoc = goldDBName
        
        GoldDB.goldDB = {}
        GoldDB.countries = set()

        with open(GoldDB.goldDBFileLoc, 'r') as br:
            line = br.readLine()
            while line != None:
                parts = line.split("\\t")
                if parts.length == 3:
                    rel = RelationMetadata.getShortenedRelation(parts[2])
                    GoldDB.countries.add(parts[0])
                    value = float(parts[1])
                    entityRel = (parts[0], rel)
                    if entityRel in GoldDB.goldDB:
                        GoldDB.goldDB.get(entityRel).add(0, value) #insert at front, so that latest values are in front.
                    else:
                        valueList = []
                        valueList.append(value)
                        GoldDB.goldDB[entityRel] = valueList
                
                line = br.readLine()

            br.close()
        

    '''
    function returns list of gold values for entity, relation pair
    '''
    @staticmethod
    def getGoldDBValue(entity, rel , k = None):
        if k == None:
            entityRel = (entity, rel)
            if entityRel in GoldDB.goldDB:
                return GoldDB.goldDB[entityRel]
            return None
        else:
            entityRel = (entity, rel)
            if entityRel in GoldDB.goldDB:
                resultList = GoldDB.goldDB[entityRel]
                max_val = (k < resultList.length) if k else resultList.length
                return list(resultList[0, max_val])
            return []
    
        
    '''
    function returns top gold value for entity, relation pair
    '''
    @staticmethod
    def getTopGoldDBValue(entity, rel):
        entityRel = (entity, rel)
        if entityRel in GoldDB.goldDB:
            return GoldDB.goldDB[entityRel][0]
        return None
    
    @staticmethod
    def getCountries():
        return GoldDB.countries

class PruneFalseInstances:

    def __init__(self):
        self.gdb = None
        
    def PruneFalseInstances(self):
        self.gdb = GoldDB()
    
    
    def run(self, instancesFile, prunedInstancesFile):
        
        with open(instancesFile,'r','w') as f:
        
            instanceLine = None
            linesProcessed = 0
            bugs = 0
            instanceLine = f.readLine()
            while instanceLine != None:
                instanceLineSplit = instanceLine.split("\t")
                linesProcessed += 1
                location = instanceLineSplit[0]
                number = float(instanceLineSplit[4])
                if number == None:
                    print("No legal conversion for " + instanceLineSplit[4])
                    bugs += 1
                    continue
                
                relation  = instanceLineSplit[9]
                if alg.lpercp.GoldDbInference.closeEnough(number, relation, location):
                    f.write(instanceLine + "\n")
                
                
                if linesProcessed % 100000 == 0:
                    print("Lines Processed: " + linesProcessed)
                
                instanceLine = f.readLine()
                
        f.close()
        print("Total bugs: " + bugs)
 
'''
 * This class iterates over a knowledge base and determines which of the
 * location relations are confused. Confused: For a given location, relations r1
 * and r2 are said to be confused if r1 and r2 have the same units and the range
 * of true values of r1 and r2 are similar
 '''
class ConfusedLocationUnits:

    gdb = GoldDB()

    '''
     * checks if a = b +- k*b
     * 
     * @param a
     * @param b
     * @param k
     *            , a fraction
     * @return
     '''
    def withinKPercent(self, a, b, k):
        assert (k <= 1 and k >= 0)
        return (a >= b - b * k and a <= b + b * k)

    def sum(self, lst):
        res = float(0)
        for d in lst:
            res += d
        
        return res

    def mean(self, lst):
        assert (list.length > 0)
        return sum(list) / list.length

    '''
     * Determines whether rel1 and rel2 for the given country can be confused
     * 
     * @param country
     * @param rel1
     * @param rel2
     * @return
    '''
    def isConfused(self, country, rel1, rel2):
        rel1Vals = GoldDB.getGoldDBValue(country, rel1)
        rel2Vals = GoldDB.getGoldDBValue(country, rel2)
        
        if(None == rel1Vals or None == rel2Vals):
            return False
        
        closeness_criteria = float(0.20)
        return self.withinKPercent(self.mean(rel1Vals), self.mean(rel2Vals), closeness_criteria)
