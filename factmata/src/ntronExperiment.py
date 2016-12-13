'''
Created on 13.12.16

@author: Yunus Emrah Bulut
'''
import os
from numberTron.datastructures.graph import Graph
from numberTron.utils.jasonUtils import JsonUtils

class NtronExperiment:
    
    # Feature thresholds
    MINTZ_FEATURE_THRESHOLD = None
    KEYWORD_FEATURE_THRESHOLD = None
    
    
    def __init__(self, propertiesFile):
        
        self.corpusPath
        self.mintzKeywordsFg, self.numberFg, self.keywordsFg
        self.sigs
        self.DSFiles
        self.featureFiles
        self.ntronModelDirs
        self.cis

        self.countriesFile
        self.useKeywordFeatures = False
        self.rbased
        self.countryFreebaseIdMap

        # Properties for the averaged perceptron
        self.regularizer # regularizer to dampen the weights
        self.numIterations # the number of iterations of the perceptron
        self.finalAvg # should the parameters be finally divided?
    
        # Gold database matching params
        self.topKGoldDb # match the recent k params from the gold database
        self.MARGIN       
        
        # confused relations ignoring
        self.ignoreConfusion

        properties = JsonUtils.getJsonMap(propertiesFile)

        rbased = new RuleBasedDriver(True)
        corpusPath = JsonUtils.getStringProperty(properties, "corpusPath")

        # Create the entity name to id map

        countriesFile = JsonUtils.getStringProperty(properties, "countriesList")

        with open(countriesFile, 'r') as br:
            
            countryRecord = None
            countryFreebaseIdMap = {}
            countryRecord = br.readLine()
            while countryRecord != None:
                vars = countryRecord.split("\t")
                if len(vars) == 2:
                    countryName = vars[1].lower()
                    countryId = vars[0];
                    countryFreebaseIdMap[countryName] = countryId
                    countryRecord = br.readLine()

            br.close()
        
        # end creating the map
         
        useKeywordFeature = JsonUtils.getStringProperty(properties, "useKeywordFeatures")
        if useKeywordFeature != None:
            if useKeywordFeature == "true":
                self.useKeywordFeatures = True

        keywordFeatureGeneratorClass = JsonUtils.getStringProperty(properties, "keywordsFg")
        mintzFeatureGeneratorClass = JsonUtils.getStringProperty(properties, "mintzKeywordsFg")
        numbersFeatureGeneratorClass = JsonUtils.getStringProperty(properties, "numbersFg")

        if keywordFeatureGeneratorClass != None and (not len(keywordFeatureGeneratorClass) == 0):
            @ TODO
            self.keywordsFg = ClassLoader.getSystemClassLoader().loadClass(keywordFeatureGeneratorClass).newInstance()
        
        if mintzFeatureGeneratorClass != None and (not len(mintzFeatureGeneratorClass) == 0):
            self.mintzKeywordsFg = ClassLoader.getSystemClassLoader().loadClass(mintzFeatureGeneratorClass).newInstance()

    
        if (numbersFeatureGeneratorClass != None and (not len(numbersFeatureGeneratorClass == 0):
            self.numberFg = ClassLoader.getSystemClassLoader().loadClass(numbersFeatureGeneratorClass).newInstance()

        sigClasses = JsonUtils.getListProperty(properties, "sigs")
        sigs = []
        
        for sigClass in sigClasses:
            sigs.append(ClassLoader.getSystemClassLoader().loadClass(sigClass).getMethod("getInstance").invoke(null))

        dsFileNames = JsonUtils.getListProperty(properties, "dsFiles")
        DSFiles = []
        for dsFileName in dsFileNames:
            DSFiles.append(dsFileName)

        featureFileNames = JsonUtils.getListProperty(properties,"featureFiles")
        featureFiles = []
        for featureFileName in featureFileNames:
            featureFiles.add(featureFileName)

        ntronModelDirs = []
        multirDirNames = JsonUtils.getListProperty(properties, "models")
        for multirDirName in multirDirNames:
            ntronModelDirs.append(multirDirName)

        cis = CustomCorpusInformationSpecification()

        altCisString = JsonUtils.getStringProperty(properties, "cis")
        if altCisString != None:
            cis = ClassLoader.getSystemClassLoader().loadClass(altCisString).newInstance()

        # CorpusInformationSpecification
        tokenInformationClassNames = JsonUtils.getListProperty(properties, "ti")
        tokenInfoList = []
        for tokenInformationClassName in tokenInformationClassNames:
            tokenInfoList.append(ClassLoader.getSystemClassLoader().loadClass(tokenInformationClassName).newInstance())

        sentInformationClassNames = JsonUtils.getListProperty(properties, "si")
        sentInfoList = []
        for sentInformationClassName in sentInformationClassNames:
            sentInfoList.append(ClassLoader.getSystemClassLoader().loadClass(sentInformationClassName).newInstance())

        docInformationClassNames = JsonUtils.getListProperty(properties, "di")
        docInfoList = []
        for docInformationClassName in docInformationClassNames:
            docInfoList.append(ClassLoader.getSystemClassLoader().loadClass(docInformationClassName).newInstance())

        ccis = cis
        ccis.addDocumentInformation(docInfoList)
        ccis.addTokenInformation(tokenInfoList)
        ccis.addSentenceInformation(sentInfoList)

        # perceptron params setup
        self.regularizer = JsonUtils.getDoubleProperty(properties, "regularizer")
        self.numIterations = JsonUtils.getIntegerProperty(properties, "iterations")
        self.finalAvg = JsonUtils.getBooleanProperty(properties, "finalAvg")

        # gold db params setup

        goldDBFileLoc = JsonUtils.getStringProperty(properties, "kbRelFile")

        self.topKGoldDb = JsonUtils.getIntegerProperty(properties, "topKGoldDb")
        self.MARGIN = JsonUtils.getDoubleProperty(properties, "margin")
        GoldDB.initializeGoldDB(goldDBFileLoc, topKGoldDb, MARGIN)

        # Feature thresholds
        NtronExperiment.KEYWORD_FEATURE_THRESHOLD = JsonUtils.getIntegerProperty(properties, "keywordFeatureThreshold")
        NtronExperiment.MINTZ_FEATURE_THRESHOLD = JsonUtils.getIntegerProperty(properties, "mintzFeatureThreshold")

        # Confused relations ignore
        self.ignoreConfusion = JsonUtils.getBooleanProperty(properties, "ignoreConfusion")
    
    '''
     * The orchestrator. Runs spotting, preprocessing, feature generation and
     * training in this order. Only starts steps that are needed
     * 
    '''
        
    def run(self):
        
        c = Corpus(corpusPath, cis, true)
        
        # Step 1: create a file of all the possible spots
        runDS = not filesExist(DSFiles)
        
        if runDS:
            print("Running DS")
            spotting = UnitLocationSpotting(corpusPath, cis, rbased, countriesFile)
            spotting.iterateAndSpot(DSFiles[0], c)
        

        # Step 2: Generate features 
        runFG = not filesExist(featureFiles)
        if runFG:
            print("Running Feature Generation")
            fGeneration = NumbertronFeatureGenerationDriver(mintzKeywordsFg, numberFg, keywordsFg)
            fGeneration.run(DSFiles, featureFiles, c, cis)
        

        print("Training " + ntronModelDirs[0])
        # Step 3: Training and weight learning
        # Step 3.1: From the feature file, generate graphs
        for i in range(featureFiles):
            with open(ntronModelDirs[i]) as modelFile:
                directory = os.path.dirname(modelFile)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                MakeGraph.run(featureFiles[i], ntronModelDirs[0]+ os.path.sep + "mapping", 
                              ntronModelDirs[0]+ os.path.sep + "train", ntronModelDirs[0])
        
        with open(ntronModelDirs[0]) as modelFile:
            #**Print Graph*
            train = LRGraphMemoryDataset(ntronModelDirs[0] + os.path.sep + "train")
            lrg = Graph()
            with open('graph.txt','w') as bw:
                    while train.next(lrg):
                        bw.write(lrg.toString() + "\n")
                        bw.write("\n\n")
                
            bw.close()
        
            LperceptTrain.train(modelFile.getAbsoluteFile().toString(), new Random(
                    1), this.numIterations, this.regularizer, this.finalAvg,
                    this.ignoreConfusion, ntronModelDirs.get(0)
                            + File.separatorChar + "mapping")

    def filesExist(self, dsFiles):
        for s in dsFiles:
            if not os.path.isfile(s):
                print(s + " File does not exist!Need To Generate it")
                return False
            
        return True

    '''
     * Just a meta function to facilitate debugging. Creates a fairly large
     * feature weight file for each for the relations.
     * 
     * @param mapping
     * @param parametersFile
     * @param modelFile
     * @param outFile
    '''
   @staticmethod
    def writeFeatureWeights(mapping, parametersFile, modelFile, outFile):
        
        bw = open(outFile, 'w')
        featureReader = open(mapping, 'r')
        numRel = int(featureReader.readLine())
        relNumNameMapping = {}

        for i in range(numRel):
            # skip relation names
            relNumNameMapping[i] = featureReader.readLine()
    
        numFeatures = int(featureReader.readLine())
        ftr = None
        featureList = {}
        fno = 0
        while fno < numFeatures:
            ftr = featureReader.readLine()
            featureList[fno] = ftr
            fno += 1
    
        p = Parameters()
        p.model = Model()
        p.model.read(modelFile)
        p.deserialize(parametersFile)
        for r in range(p.model.numRelations):
            relName = relNumNameMapping[r]
            dv = p.relParameters[r]
            for i in range(numFeatures):
                bw.write(relName + "\t" + str(featureList[i]) + "\t" + str(dv.vals[i]) + "\n")
        bw.close()
        featureReader.close()

    '''
     * 
     * @param iterations
     *            Iterations of the perceptron
     * @param regularizer
     *            Damper
     * @param topk
     *            Match the top k values in the gold db
     * @param margin
     *            match margin
     * @param mintzFeatureThreshold
     * @param kwFeatureThreshold
     * @param finalAvg
     *            Should the parameters be averaged?
     * @param extractionCutoff
    '''
    def updateHyperparams(self, iterations, regularizer, topk, margin, mintzFeatureThreshold, kwFeatureThreshold, finalAvg):
        self.numIterations = iterations
        self.regularizer = regularizer
        self.topKGoldDb = topk
        NtronExperiment.MINTZ_FEATURE_THRESHOLD = mintzFeatureThreshold
        NtronExperiment.KEYWORD_FEATURE_THRESHOLD = kwFeatureThreshold
        self.finalAvg = finalAvg


if __name__=='__main__':

        irb = NtronExperiment(xxx)
        irb.run()
        # pw = open("hitstats", 'w')
        # GoldDbInference.printMatchStats(pw)
        # pw.close()
        '''
         self.writeFeatureWeights(irb.ntronModelDirs[0] + os.path.sep
         + "mapping", irb.ntronModelDirs[0] + os.path.sep
         + "params", irb.ntronModelDirs[0] + os.path.sep
         + "model", irb.ntronModelDirs[0] + os.path.sep
         + "weights");
        '''

    