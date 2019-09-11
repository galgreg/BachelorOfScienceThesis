import json
from os import remove
from train import *
from ddt import ddt, data, unpack
import unittest
from unittest.mock import patch

@ddt
class TestTrain(unittest.TestCase):
    @data(None, 1.1, 13, [1, 2, 3, 4], {"invalid" : "path"}, "", \
            "NON_EXISTENT_FILE.json")
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
            (None, 3, {}), (5, None, {}), (None, None, {}),
            (3.14, 5, {}), ([1, 2, 3], 5, {}), ({3 : 14}, 5, {}), (5, 3.14, {}),
            (5, [3, 14], {}), (5, {3 : 14}, {}), (2, 2, "kanapka"), (2, 2, 2),
            (2, 2, [1, 2, 3]), (2, 2, 3.14))
    def test_computeAgentDimensions_SomeParametersHaveWrongType(
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

    @data(1, 3.14, (1, 2, 3), {"wrong" : "type"}, "kanapka", None)
    def test_findIndexOfBestModel_FitnessListHasWrongType(self, fitnessList):
        self.assertRaises(
                ValueError,
                findIndexOfBestModel,
                fitnessList)
    
    @unpack
    @data(([0.3, 1.1, 5.9, 3.6], 2), ([-100.13, -2.3, -5.7, -0.9], 3),
            ([-100.13, 0.5, -5.7, -0.9], 1))
    def test_findIndexOfBestModel_OK(self, fitnessList, expectedIndex):
        actualIndex = findIndexOfBestModel(fitnessList)
        self.assertEqual(actualIndex, expectedIndex)

    @data(1, 3.14, "kanapka", (3, 14), {"wrong" : "type"}, None)
    def test_areAllAgentsDone_AgentDonesHasWrongType(self, agentDones):
        self.assertRaises(
                ValueError,
                areAllAgentsDone,
                agentDones)
    
    def test_areAllAgentsDone_SomeElementsAreNotBools(self):
        agentDones = [True, "kanapka", 3.14, False, {"wrong" : "type"}, True]
        self.assertRaises(
                ValueError,
                areAllAgentsDone,
                agentDones)

    @unpack
    @data(([True, True, False, True], False), ([True, True, True, True], True))
    def test_areAllAgentsDone_False(self, agentDones, expectedResult):
        actualResult = areAllAgentsDone(agentDones)
        self.assertEqual(actualResult, expectedResult)
    
    @data(1, 3.14, "kanapka", {"wrong" : "type"}, None, (1, 2, 3, 4))
    def test_getBestFitness_FitnessListHasWrongType(self, fitnessList):
        self.assertRaises(
                ValueError,
                getBestFitness,
                fitnessList)
    
    def test_getBestFitness_FitnessListIsEmpty(self):
        fitnessList = []
        self.assertRaises(
                ValueError,
                getBestFitness,
                fitnessList)

    @unpack
    @data(([-1.3, 2.0, 10.13, -4.1], 10.13), ([-10.9, -2.0, -0.9, -33.3], -0.9))
    def test_getBestFitness_OK(self, fitnessList, expectedValue):
        actualValue = getBestFitness(fitnessList)
        self.assertEqual(actualValue, expectedValue)
