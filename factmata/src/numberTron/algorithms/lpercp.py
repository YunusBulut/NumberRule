'''
Created on 10.12.16
@author: Yunus Emrah Bulut

This module mimics the lpercp algorithm classes of NEO-IE/numbertron.
@link: https://github.com/NEO-IE/numbertron/tree/master/src/main/java/iitb/neo/training/algorithm/lpercp

'''
import numberTron.utils.keywordData as kwd
from nltk.stem.snowball import SnowballStemmer
import numberTron.datastructures.graph as gr
import math
import operator
from numberTron.utils import mathUtils, randomUtils
from numberTron.golddb import goldDB


class Parse:
    
    def __init__(self):
        self.graph
        self.n_states
        self.z_states

class DenseVector:
    
    def __init__(self, length):
        
        self.vals = [None]*length

    def dotProduct(self, sbv, denseVector = None, featureScoreMap = None):
        
        summmation = float(0)
        if denseVector != None and featureScoreMap == None:
            for i in range(sbv.num):
                featureScore = float(denseVector.vals[sbv.ids[i]])
                summmation = summmation + featureScore
        elif denseVector == None and featureScoreMap == None:
            for i in range(sbv.num):
                featureScore = self.vals[sbv.ids[i]]
                summmation = summmation + featureScore
        else:
            for i in range(sbv.num):
                featureScore = denseVector.vals[sbv.ids[i]]
                featureScoreMap[int(sbv.ids[i])] = float(featureScore)
                summmation = summmation + featureScore
        
        return summmation

    def reset(self):
        
        self.vals = [None]*self.vals.length

    def copy(self):

        n = list(self.vals)
        return n

    def scale(self, factor):
        
        self.vals[:] = [x * 2 for x in self.vals]
        
    def addSparse(self, sbv, factor):
        
        for i in range(sbv.num):
            self.vals[sbv.ids[i]] += float(factor)
            
    def sum(self, denseVector1, denseVector2, factor):
        n = DenseVector(denseVector1.vals.length)
        for i in range(denseVector1.vals.length):
            n.vals[i] = denseVector1.vals[i] + float(factor)*denseVector2.vals[i]
        return n  
    
class Model:
    
    def __init__(self):
        self.numFeaturesPerRelation
        self.noRelationState
    
    def numFeatures(self, rel):
        return self.numFeaturesPerRelation[rel]
    
    def read(self, fileName):
        with open(fileName, 'r', encoding='utf-8') as f:
            numRelations = int(f.readline());
            numFeaturesPerRelation = [None]*numRelations
            for i in range(numRelations):
                numFeaturesPerRelation[i] = int(f.readLine())
        f.close()
    
    def write(self, fileName):
        with open(fileName, 'w', encoding='utf-8') as f:
            f.write(self.numRelations + "\n");
            for i in range(self.numFeaturesPerRelation.length):
                f.write(self.numFeaturesPerRelation[i] + "\n");
        f.close()

class Parameters:
    
    def __init__(self):
        self.relParameters
        self.model
    
    def sum(self, denseVector1, denseVector2, factor, parameters = None):
        
        if parameters == None:
            if denseVector1 == None and denseVector2 == None:
                return None
            elif denseVector2 == None:
                return denseVector1.copy()
            elif denseVector1 == None:
                v = denseVector2.copy()
                v.scale(factor)
                return v
            else:
                return denseVector1.sum(denseVector2, factor)
        else:
            for i in range(self.relParameters.length):
                self.relParameters[i] = sum(self.relParameters[i], parameters.relParameters[i], factor, None)
    
    def init(self):
        if self.relParameters == None:
            self.relParameters = DenseVector[self.model.numRelations]
            print("requesting " + (8*self.relParameters.length*int(self.model.numFeaturesPerRelation[0])) + " bytes")
            for i in range(self.relParameters.length):
                self.relParameters[i] = DenseVector(self.model.numFeatures(i))
    def reset(self):
        for i in range(self.relParameters.length):
            if self.relParameters[i] == None:
                self.relParameters[i].reset()     

class Scorer:
    
    def __init__(self):
        self.Params
    
    def scoreMentionRelation(self, doc, m, rel, featureScoreMap = None):
        summmation = float(0)
        p = self.params.relParameters[rel]
        if featureScoreMap != None:
            summmation += p.dotProduct(doc.features[m], featureScoreMap)
        else:
            summmation += p.dotProduct(doc.features[m])
        return summmation
    
    def getMentionRelationFeatures(self, doc, m, rel):
        return doc.features[m]
    
    
    def getMentionNumRelationFeatures(self, doc, m, rel):
        return doc.numFeatures[m]
    
    
    def setParameters(self, params):
        self.params = params

class KeywordInference:
    
    marginMap = {}
    
    @staticmethod   
    def hasKeyword(feats, rel):
        relKey = kwd.KeywordData.REL_KEYWORD_MAP.get(rel)
        stemmer = SnowballStemmer("english")
        for key in relKey:
            stemKey = stemmer.stem(key.toLowerCase());
            featID = LocalAveragedPerceptron.featNameNumMapping.get("key: "+stemKey);
            if featID != None:
                if feats.contains(featID):
                    return True
                
        return False
    
    @staticmethod
    def infer(grph):
        p = Parse()
        p.graph = grph
        p.z_states = [False]*grph.Z.length
        p.n_states = [False]*grph.n.length
        numN = grph.n.length
                
        for n_i in range(numN):
            feats = set()
            n = grph.n[n_i]
            z_s = n.zs_linked
            for z in z_s:
                for i in grph.features[z].ids:
                    feats.add(i)
                
            
            #Make the number true if one of the Z nodes attached expresses the relation
            p.n_states[n_i] = KeywordInference.hasKeyword(feats, grph.relation)
        
        return p
    
    
 
class LocalAveragedPerceptron:
    
    relNumNameMapping = None
    featNameNumMapping = None
        
    def __init__(self, model, random, maxIterations, regularizer, finalAverageCalc, mappingFile): 
        
        self.numIterations = maxIterations
        self.computeAvgParameters = True
        self.finalAverageCalc = finalAverageCalc
        self.delta = 1
        self.regulaizer = regularizer
        self.scorer = Scorer()
        self.model = model
        self.random = random 
        '''
        variables added for debugging purposes.
        '''        
        self.featureList
        self.numRelation
        self.numFeatures
        self.outputFile = "verbose_iteration_updates_key_area_1"
        self.obw
        self.debug = False
        self.readMapping = True
        self.numDisagreements = 0
        
        '''
        the following two are actually not storing weights:
        the first is storing the iteration in which the average weights were
        last updated, and the other is storing the next update value
        '''
        self.avgParamsLastUpdatesIter
        self.avgParamsLastUpdates
        
        '''
        The following parameter array stores the number of times a particular
        parameter
        has been updated, used for regularization in some sense.
        '''
        self.lastZeroIter
        self.countZeroIter
    
        self.avgParameters
        self.iterParameters
        
        '''
        The following parameter array stores the number of times a particular 
        parameter was updated, this will help in smoothing weights that are updated a lot (well that's the hope)
        '''
        self.countUpdates
        self.avgIteration = 0
  
        if self.readMapping:
            relNumNameMapping = {}
            featNameNumMapping = {}
            featureList = {}
            with open(mappingFile, 'r') as featureReader:
                numRel = int(featureReader.readLine())
                for i in range(numRel):
                    rel = featureReader.readLine().trim()
                    relNumNameMapping[i] = rel
                
                numFeatures = int(featureReader.readLine())
                ftr = None
                featureList = {}
                fno = 0
                while fno < numFeatures:
                    ftr = featureReader.readLine().trim()
                    parts = ftr.split("\t")
                    featNameNumMapping[parts[1]] = int(parts[0])
                    featureList[fno] = ftr
                    fno += 1
                
                featureReader.close()
  
    def trainingIteration(self, iteration, trainingData):

        lrg = gr.Graph()
        trainingData.shuffle(self.random)
        trainingData.reset()
        
        while trainingData.next(lrg):
            if lrg.features.length == 0:
                continue
            # compute most likely label under current parameters
            predictedParse = FullInference.infer(lrg, self.scorer, self.iterParameters)
            trueParse = ConditionalInference.infer(lrg, self.scorer, self.iterParameters)

            if not self.NsAgree(predictedParse, trueParse):
                self.numDisagreements += 1
                # if this is the first avgIteration, then we need to initialize
                # the lastUpdate vector
                if self.computeAvgParameters and self.avgIteration == 0:
                    self.avgParamsLastUpdates.sum(self.iterParameters, float(1.0))
                

                self.update(predictedParse, trueParse)
            

            if self.computeAvgParameters:
                self.avgIteration += 1
            
    
    def update(self, predictedParse, trueParse):
            
        # if this is the first avgIteration, then we need to initialize
        # the lastUpdate vector
        if self.computeAvgParameters and self.avgIteration == 0:
            self.avgParamsLastUpdates.sum(self.iterParameters, float(1.0))
        lrg = predictedParse.graph

        numMentions = lrg.numMentions
        for i in range(numMentions):
            
            v1a = self.scorer.getMentionRelationFeatures(lrg, i, lrg.relNumber)
            if trueParse.z_states[i] == True:
                self.updateRel(lrg.relNumber, v1a, self.delta, self.computeAvgParameters)

            
            if predictedParse.z_states[i] == True:
                self.updateRel(lrg.relNumber, v1a, -self.delta, self.computeAvgParameters)
            
    def NsAgree(self, predictedParse, trueParse):
        
        numN = predictedParse.n_states.length
        if numN != trueParse.n_states.length:
            raise ValueError("Something is not right in LocalAveragedPerceptron")
                    
        
        for i in range(numN):
            if predictedParse.n_states[i] != trueParse.n_states[i]:
                return False
            
        
        return True
    
    def updateRel(self, relNumber, features, delta, useIterAverage):

        self.iterParameters.relParameters[relNumber].addSparse(features, delta)
        
        # updating numeric features.
         
        # useIterAverage = false;
        if useIterAverage:

            lastUpdatesIter = self.avgParamsLastUpdatesIter.relParameters[relNumber]
            lastUpdates = self.avgParamsLastUpdates.relParameters[relNumber]
            avg = self.avgParameters.relParameters[relNumber]
            iter = self.iterParameters.relParameters[relNumber]

            lastZeroIteration = self.lastZeroIter.relParameters[relNumber]
            zeroIterationCount = self.countZeroIter.relParameters[relNumber]
            
            updateCountVector = self.countUpdates.relParameters[relNumber]
            
            for j in range(features.num):
                id = features.ids[j]
                updateCountVector.vals[id] += 1
                if lastUpdates.vals[id] != 0:
                                 
                    notUpdatedWindow = self.avgIteration - int(lastUpdatesIter.vals[id])
                    avg.vals[id] = math.pow(self.regulaizer, notUpdatedWindow)* avg.vals[id] + notUpdatedWindow* lastUpdates.vals[id]

                    if iter.vals[id] == 0:
                        assert (lastZeroIteration.vals[id] == -1)
                        lastZeroIteration.vals[id] = self.avgIteration
                    elif lastZeroIteration.vals[id] != -1:
                        zeroIterationCount.vals[id] += (self.avgIteration - lastZeroIteration.vals[id])
                        lastZeroIteration.vals[id] = -1
                    

                    if self.debug:
                        if id == 527682:
                            self.obw.write("\n" + self.relNumNameMapping.get(relNumber) + "--> " + delta + "\n")
                            self.obw.write(lastUpdatesIter.vals[id] + "-->" + self.avgIteration + "\n")
                            self.obw.write(self.featureList.get(id) + " : " + avg.vals[id] + "\n")
                            self.obw.write("Iterval : " + iter.vals[id] + "\n")
                            self.obw.write("*************************************\n")
                        
                    
                
                lastUpdatesIter.vals[id] = self.avgIteration
                lastUpdates.vals[id] = iter.vals[id]
            
        
    def finalizeRel(self):
        
        for s in range(self.model.numRelations):
            lastUpdatesIter = self.avgParamsLastUpdatesIter.relParameters[s]
            lastUpdates = self.avgParamsLastUpdates.relParameters[s]
            avg = self.avgParameters.relParameters[s]
            zeroIterationCountRel = self.countZeroIter.relParameters[s]
            
            updateCountVector = self.countUpdates.relParameters[s] 
            
            for idd in range(avg.vals.length):
                if lastUpdates.vals[idd] != 0:
                    notUpdatedWindow = self.avgIteration - int(lastUpdatesIter.vals[idd])
                    avg.vals[idd] = math.pow(self.regulaizer, notUpdatedWindow)* avg.vals[idd] + notUpdatedWindow* lastUpdates.vals[idd]

                    nonZeroIteration = self.avgIteration - int(zeroIterationCountRel.vals[idd])
                    
                    if self.finalAverageCalc:
                        avg.vals[idd] = updateCountVector.vals[idd] == 0 if avg.vals[idd] else (avg.vals[idd] / updateCountVector.vals[idd])
                    
                    lastUpdatesIter.vals[idd] = self.avgIteration

    def train(self, trainingData):

        if self.computeAvgParameters:
            avgParameters = Parameters()
            avgParameters.model = self.model
            avgParameters.init()

            avgParamsLastUpdatesIter = Parameters()
            avgParamsLastUpdates = Parameters()
            avgParamsLastUpdatesIter.model = avgParamsLastUpdates.model = self.model
            avgParamsLastUpdatesIter.init()
            avgParamsLastUpdates.init()

            lastZeroIter = Parameters()
            lastZeroIter.model = self.model
            lastZeroIter.init()

            countZeroIter = Parameters()
            countZeroIter.model = self.model
            countZeroIter.init()
            

            countUpdates = Parameters()
            countUpdates.model = self.model
            countUpdates.init()

        
        iterParameters = Parameters()
        iterParameters.model = self.model
        iterParameters.init()

        for i in range(self.numIterations):
            if self.debug:
                with open(self.outputFile, 'w') as obw:
                    obw.write("#######################################\n");
                    obw.write("Iteration : " + i + "\n");
                    obw.write("#######################################\n");
            print("Iteration: " + i)
            self.trainingIteration(i, trainingData)
            print("Disagreements: " + self.numDisagreements)
            self.numDisagreements = 0
            
        if self.computeAvgParameters:
            self.finalizeRel()

        if self.debug:
            obw.close()
            
        return (self.computeAvgParameters) if self.avgParameters else self.iterParameters
    
class FullInference:
    
    @staticmethod
    def infer(lrg, scorer, params):
        
        p = Parse()
        #setup what we already know about the parse 
        p.graph = lrg
        scorer.setParameters(params)
        p.z_states = [False]*lrg.Z.length
        p.n_states = [False]*lrg.n.length
        #iterate over the Z nodes and set them to true whenever applicable
        numZ = lrg.Z.length
        for z in range(numZ):
            bestScore = float(0.0)

            #There can be multiple "best" relations. It is okay if we get anyone of them
            bestRels = []
            for r in range(params.model.numRelations):
                currScore = scorer.scoreMentionRelation(lrg, z, r)
                if currScore > bestScore:
                    bestRels.clear()
                    bestRels.add(r)
                    bestScore = currScore
                elif bestScore > 0 and currScore == bestScore:
                    bestRels.add(r)
                
            if bestRels.contains(lrg.relNumber):
                p.z_states[z] = True
            else:
                p.z_states[z] = False
            
        LEAST_Z_FLIPPED_COUNT = 0.5
        #now flip n nodes accordingly: OR 
        numN = lrg.n.length
        for n_i in range(numN):
            attachedZ = lrg.n[n_i].zs_linked
            totalZ = attachedZ.size()
            p.n_states[n_i] = False
            trueAttachedZCount = 0
            for z in attachedZ: #iterate over all the attached Z nodes
                if p.z_states[z]: # if any of them is one, set the number node to 1
                    trueAttachedZCount +=1
                    p.n_states[n_i] = (((trueAttachedZCount * 1.0) / (totalZ)) >= LEAST_Z_FLIPPED_COUNT)
               
        return p
    
    '''
     * Returns a map with score of all the relations
     * 
     * @param lrg
     * @param scorer
     * @param params
     * @return
     '''
    @staticmethod
    def getRelationScoresPerMention(lrg, scorer, params):
        p = Parse()
        #setup what we already know about the parse
        p.graph = lrg
        scorer.setParameters(params)
        p.z_states = [False]*lrg.Z.length

        #iterate over the Z nodes and set them to true whenever applicable
        numZ = lrg.Z.length
        assert (numZ == 1)
        relationScoreMap = {}
        for r in range(params.model.numRelations):
            relationScoreMap[r] = scorer.scoreMentionRelation(lrg, 0, r)
        
        return  sorted(relationScoreMap.items(), key=operator.itemgetter(1))

class ConditionalInference:
    
    @staticmethod
    def infer(lrg, scorer, params):
                
            p = Parse()
            p.graph = lrg
            scorer.setParameters(params)
            
            trueParse = KeywordInference.infer(lrg)
           
            p.z_states = [False]*lrg.Z.length
            p.n_states = trueParse.n_states
            
            for i in range(lrg.n.length):
                n = lrg.n[i]
                z_s = n.zs_linked
                for z in z_s:
                    p.z_states[z] = trueParse.n_states[i]  #z_s copy the state of n_s.
                
            return p

'''
 * Assigns a binary label to all the nodes given the current weights;
 *For numbers, we want the one-labeled nodes to be proximal.
 * In this scheme we start with the \textbf{Atleast-K} assignment 
 * (call it $\bar{\vn}$) and set to zero any $n^r_q$ outside a range of
 *  $\pm \delta_r\%$ of a chosen central value.  We choose the central value 
 *  $c$ for which $\bar{n}_c^r=1$ and which causes smallest number of
 *   $\bar{n}_q^r=1$ to set to zero. 

'''
class FullInferenceAgreeingK:
    
    @staticmethod
    def infer(lrg, scorer, params):
        p = Parse()
        #setup what we already know about the parse
        p.graph = lrg
        scorer.setParameters(params)
        p.z_states = [False]*lrg.Z.length
        p.n_states = []*lrg.n.length
        #iterate over the Z nodes and set them to true whenever applicable
        numZ = lrg.Z.length
        for z in range(numZ):
            bestScore = float(0.0)

            #There can be multiple "best" relations. It is okay if we get anyone of them
            bestRels = []
            for r in range(params.model.numRelations):
                currScore = scorer.scoreMentionRelation(lrg, z, r)
                if currScore > bestScore:
                    bestRels.clear()
                    bestRels.add(r)
                    bestScore = currScore
                elif bestScore > 0 and currScore == bestScore:
                    bestRels.add(r)
            
            if bestRels.contains(lrg.relNumber):
                p.z_states[z] = True
            else:
                p.z_states[z] = False
        
        LEAST_Z_FLIPPED_COUNT = float(0.5)
        
        #Now get the Atleast-K assignments to the number nodes
        numN = lrg.n.length
        for n_i in range(numN):
            attachedZ = lrg.n[n_i].zs_linked
            
            totalZ = attachedZ.size()
            p.n_states[n_i] = False
            trueAttachedZCount = 0
            for z in attachedZ: # iterate over all the attached Z nodes
                if p.z_states[z]: # if any of them is one, set the number node to 1
                    trueAttachedZCount += 1
                    p.n_states[n_i] = (((trueAttachedZCount * 1.0) / (totalZ)) >= LEAST_Z_FLIPPED_COUNT)
        
        # Now start the agreeing k
        #Different deltas for different relations
        delta = GoldDbInference.marginMap.get(lrg.relation)
        leastFlips = int(2**31 - 1)
        bestCentralValue = float(-1)
        
        #Find the optimal central value
        for n_i in range(numN):
            if p.n_states[n_i]: #potential central value?
                flipsCaused = 0;
                centralValue = lrg.n[n_i].value
                for n_c in range(numN):
                    if n_c == n_i:
                        continue
                    if p.n_states[n_c] and not (mathUtils.within(lrg.n[n_c].value, centralValue, delta)): #check if this guy must be turned off
                        flipsCaused +=1

                if flipsCaused < leastFlips:
                    leastFlips = flipsCaused
                    bestCentralValue = centralValue
                    
        assert(bestCentralValue != -1)
        
        #complete by actually flipping the n nodes that do not agree
        for n_i in range(numN):
            if p.n_states[n_i] and  not (mathUtils.within(lrg.n[n_i].value, bestCentralValue, delta)): #check if this guy must be turned off
                p.n_states[n_i] = False
        
        #Also set to 1 those n nodes that are zero but witnin the limits
        for n_i in range(numN):
            if not p.n_states[n_i] and (mathUtils.within(lrg.n[n_i].value, bestCentralValue, delta)): #check if this guy must be turned off
                p.n_states[n_i] = True
        
        return p

    '''
     * Returns a map with score of all the relations
     * 
     * @param lrg
     * @param scorer
     * @param params
     * @return
    '''
    @staticmethod
    def getRelationScoresPerMention(lrg, scorer, params):
        p = Parse()
        #setup what we already know about the parse
        p.graph = lrg
        scorer.setParameters(params)
        p.z_states = [False]*lrg.Z.length

        #iterate over the Z nodes and set them to true whenever applicable
        numZ = lrg.Z.length
        assert (numZ == 1)
        relationScoreMap = {}
        for r in range(params.model.numRelations):
            relationScoreMap[r] = scorer.scoreMentionRelation(lrg, 0, r)
        
        return sorted(relationScoreMap.items(), key=operator.itemgetter(1))

'''
 * Sets the value of n nodes based on the value pulled from the gold db
'''
class GoldDBKeywordInference:
    
    marginMap = {}
    
    @staticmethod
    def infer(lrg):
        p = Parse()
        p.graph = lrg
        p.z_states = [False]*lrg.Z.length
        p.n_states = [False]*lrg.n.length
        numN = lrg.n.length
        for n_i in range(numN):
            feats = set()
            n = lrg.n[n_i]
            z_s = n.zs_linked
            for z in z_s:
                for idd in range(lrg.features[z].ids):
                    feats.add(idd)
                    
            if GoldDbInference.closeEnough(lrg.n[n_i].value, lrg.relation, lrg.location):
                p.n_states[n_i] = KeywordInference.hasKeyword(feats, lrg.relation)
            else:
                p.n_states[n_i] = False
                
        return p

'''
 * Sets the value of n nodes based on the value pulled from the gold db
'''

class GoldDbInference:
    
    trueCountMap = {}
    falseCountMap = {}
    xs = float(0.05)
    s = float(0.1)
    regular = float(0.2)
    xl = float(0.3)
    xxl = float(0.5)

    marginMap = {}
    marginMap["AGL"] = xl

    marginMap["FDI"] = xxl
    marginMap["GOODS"] = xl
    marginMap["GDP"] = xl
    marginMap["ELEC"] = xl
    marginMap["CO2"] = xxl
    marginMap["INF"] = s
    marginMap["INTERNET"] =xl
    marginMap["LIFE"] = regular
    marginMap["POP"] =  xl

    @staticmethod
    def infer(lrg):
        p = Parse()
        p.graph = lrg
        p.z_states = [False]*lrg.Z.length
        p.n_states = [False]*lrg.n.length
        numN = lrg.n.length
        for n_i in range(numN):
            if GoldDbInference.closeEnough(lrg.n[n_i].value, lrg.relation, lrg.location):
                p.n_states[n_i] = True
            else:
                p.n_states[n_i] = False

        return p

    '''
     * match with a specified cutoff
     * 
     * @param value
     * @param rel
     * @param entity
     * @param margin
     * @return
    '''
    @staticmethod
    def closeEnough(value, rel, entity, margin):
        bu = goldDB.GoldDB.MARGIN
        goldDB.GoldDB.MARGIN = margin
        res = GoldDbInference.closeEnough2(value, rel, entity)
        goldDB.GoldDB.MARGIN = bu
        return res
    
    @staticmethod
    def closeEnough2(value, rel, entity):
        
        locationRelation = (entity, rel)
        if rel.split("_").length > 1:
            return GoldDbInference.nullCloseEnough(value, rel, entity)
        
        rel = rel.split("&")[0]
        goldValues = goldDB.GoldDB.getGoldDBValue(entity, rel, goldDB.GoldDB.K)
        
        for val in goldValues:

            valueSlack = goldDB.GoldDB.MARGIN * val # +- 5 percent
            if (value > (val - valueSlack)) and (value < (val + valueSlack)):
                if GoldDbInference.trueCountMap[locationRelation] == None:
                    GoldDbInference.trueCountMap[locationRelation] = 1
                else:
                    currCount = GoldDbInference.trueCountMap[locationRelation]
                    GoldDbInference.trueCountMap[locationRelation] = currCount + 1
                    
                return True
        
        if GoldDbInference.falseCountMap[locationRelation] == None:
            GoldDbInference.falseCountMap[locationRelation] = 1
        else:
            currCount = GoldDbInference.falseCountMap[locationRelation]
            GoldDbInference.falseCountMap[locationRelation] = currCount + 1
    
        return False

    '''
     * This is a method that checks whether the null relation is true or not.
     * The no attachment relation is true if all the relations for which it is a
     * true class are false.
     * 
     * @param value
     * @param rel
     * @param entity
     * @return
    '''
    @staticmethod
    def nullCloseEnough(value, rel, entity):
        rels = rel.split("_")
        #for (int i = 1, l = rels.length; i < l; i++)
        for i in range(rels.length):
            i += 1
            if GoldDbInference.closeEnough2(value, rels[i], entity):
                return False
            i -= 1
        return randomUtils.RandomUtils.coinToss(0.2)
    
    '''
     * prints the stats on the number of Z nodes that are truth and false for
     * every location relation graph
    '''
    @staticmethod
    def printMatchStats(pw):
        
        for i in GoldDbInference.trueCountMap.keys():
            location = i[0]
            relation = i[1]
            hits = GoldDbInference.trueCountMap[i]
            misses = GoldDbInference.falseCountMap[i]
            if misses == None:
                misses  = 0
            
            hitPerc = None == hits if 0 else ((hits * 1.0) / (hits + misses))
            missPerc = 1 - hitPerc
            pw.write("Location = " + location + " Relation  = "
                    + relation + " hits = " + hitPerc + " misses = "
                    + missPerc + "\n")
