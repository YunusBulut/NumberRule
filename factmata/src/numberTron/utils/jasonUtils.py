'''
Created on 11.12.16

@author: Yunus Emrah Bulut
'''
import json
class JsonUtils:

    @staticmethod
    def getListProperty(properties, string):
        if string in properties:
            obj = json.loads(properties[string])
            returnValues = []
            for o in obj:
                returnValues.add(o)
            
            return returnValues
        
        return []

    @staticmethod
    def getStringProperty(properties, string):
        if string in properties:
            if properties[string] == None:
                return None
            else:
                return properties[string]
            
        return None

    
    @staticmethod
    def getDoubleProperty(properties, string):
        res = JsonUtils.getStringProperty(properties, string)
        return float(res)
    
    @staticmethod
    def getBooleanProperty(properties, str):
        res = JsonUtils.getStringProperty(properties, str)
        return res.lower() == "true"
    
    @staticmethod
    def getIntegerProperty(properties, str):
        res = JsonUtils.getStringProperty(properties, str)
        return int(res)

    
    '''
     * Returns the supplied Json file as a String key Object value map
     * @throws IOException 
     * @throws FileNotFoundException 
    '''
    
    @staticmethod
    def getJsonMap(propertiesFile):
        
        with open("../../../data/keywords.json", 'r') as f:
            json_str = f.read()
            json_data = json.loads(json_str)
            properties = json_data
        return properties
