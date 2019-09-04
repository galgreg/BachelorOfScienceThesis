import json
from os import remove
from train import loadConfigData, computeAgentDimensions
from ddt import ddt, data, unpack
import unittest

@ddt
class TestTrain(unittest.TestCase):
    @data(None, "", "NON_EXISTENT_FILE.json")
    def test_loadConfigData_InvalidFilePaths(self, invalidFilePath):
        expectedConfigData = {}
        actualConfigData = loadConfigData(invalidFilePath)
        self.assertEqual(actualConfigData, expectedConfigData)
    
    def test_loadConfigData_FileHasWrongExtension(self):
        pathWithWrongExtension = "FILE_WITH_WRONG_EXTENSION.txt"
        with open(pathWithWrongExtension, "w") as configFile:
            configFile.write('{"kanapka" : true}')
        
        expectedConfigData = {}
        actualConfigData = loadConfigData(pathWithWrongExtension)
        self.assertEqual(actualConfigData, expectedConfigData)
        remove(pathWithWrongExtension)
    
    def test_loadConfigData_EverythingIsOK(self):
        jsonContent = \
'''
{
    "zupa_rybna" : {
        "makaron" : true,
        "ile_ryb" : 5,
        "gatunki_ryb" : ["sum", "szprot", "makrela"]
    }
}
'''
        tempTestFilePath = "TEST_FILE_PATH.json"
        with open(tempTestFilePath, "w") as configFile:
            configFile.write(jsonContent)
        
        expectedConfigData = json.loads(jsonContent)
        actualConfigData = loadConfigData(tempTestFilePath)
        self.assertEqual(actualConfigData, expectedConfigData)
        remove(tempTestFilePath)
    
    @unpack
    @data((5, 3, None), (None, 3, None), (5, None, None), (None, None, None), \
            (None, 3, {}), (5, None, {}), (None, None, {}))
    def test_computeAgentDimensions_SomeParametersAreNone(
            self,
            observationSize,
            actionSize,
            configData):
        self.assertRaises(
                ValueError,
                computeAgentDimensions,
                observationSize,
                actionSize,
                configData)

    def test_computeAgentDimensions_OK(self):
        observationSize = 5
        actionSize = 3
        hiddenDimensions = [10, 30, 20]
        configData = {
            "TrainingParameters" : {
                "networkHiddenDimensions" : hiddenDimensions
            }
        }
        expectedAgentDimensions = \
                [ observationSize ] + hiddenDimensions + [ actionSize ]
        actualAgentDimensions = \
                computeAgentDimensions(observationSize, actionSize, configData)
        self.assertEqual(actualAgentDimensions, expectedAgentDimensions)
