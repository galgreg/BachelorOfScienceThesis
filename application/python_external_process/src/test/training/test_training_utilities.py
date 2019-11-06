import json
from os import remove
from src.AgentNeuralNetwork import *
from src.training.training_utilities import *
from ddt import ddt, data, unpack
import unittest
from unittest.mock import patch

@ddt
class TestTrainingUtilities(unittest.TestCase):
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
    
    def test_computeNumOfParameters(self):
        agentDimensions = [3, 5, 2]
        expectedNumOfParameters = 32
        actualNumOfParameters = computeNumOfParameters(agentDimensions)
        self.assertEqual(actualNumOfParameters, expectedNumOfParameters)
    
    @unpack
    @data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
    def test_retrieveParametersFromAgents(self, numberOfAgents, agentsDimensions):
        agentList = [
                AgentNeuralNetwork(agentsDimensions, False)
                for i in range(numberOfAgents)
        ]
        actualParameters = retrieveParametersFromAgentList(agentList)
                
        expectedParametersType = torch.Tensor
        actualParametersType = type(actualParameters)
        self.assertEqual(actualParametersType, expectedParametersType)
        
        numberOfParameters = 0
        for i in range(len(agentsDimensions) - 1):
            numberOfParameters += agentsDimensions[i] * agentsDimensions[i+1] \
                    + agentsDimensions[i+1]
        
        expectedParametersDimensions = [numberOfAgents, numberOfParameters]
        actualParametersDimensions = actualParameters.size()
        self.assertEqual(
                list(actualParametersDimensions),
                expectedParametersDimensions)
        
        expectedParameters = []
        for agent in agentList:
            expectedAgentParameters = []
            for layer in agent._layers:
                layerWeights = layer.weight
                numberOfWeights = layerWeights.numel()
                weightParameters = \
                        torch.reshape(layerWeights, (numberOfWeights,))
                expectedAgentParameters = \
                        expectedAgentParameters + weightParameters.tolist()
                expectedAgentParameters = \
                        expectedAgentParameters + layer.bias.tolist()
            
            expectedParameters.append(expectedAgentParameters)
        
        expectedParameters = torch.tensor(expectedParameters)
        self.assertTrue(torch.equal(actualParameters, expectedParameters))

    @data([5, 2], [5, 3, 2], [5, 7, 4, 2])
    def test_setNewParametersOnAgent(self, agentDimensions):
        agent = AgentNeuralNetwork(agentDimensions)
        numberOfParameters = 0
        for i in range(len(agentDimensions) - 1):
            numberOfParameters += agentDimensions[i] * agentDimensions[i+1] \
                    + agentDimensions[i+1]
        newParameters = torch.randn(numberOfParameters)
        
        parametersBeforeSet = self._retrieveParameters(agent)
        setNewParametersOnAgent(agent, newParameters)
        parametersAfterSet = self._retrieveParameters(agent)
        
        for oldLayerParameters, newLayerParameters \
                in zip(parametersBeforeSet, parametersAfterSet):
            # Assert weight and bias parameters
            for oldParameters, newParameters \
                    in zip(oldLayerParameters, newLayerParameters):
                expectedTypeOfParameters = torch.Tensor
                self.assertEqual(
                        type(oldParameters),
                        expectedTypeOfParameters)
                self.assertEqual(
                        type(newParameters),
                        expectedTypeOfParameters)
                expectedDimensionsOfParameters = \
                        list(oldParameters.size())
                actualDimensionsOfParameters = \
                        list(newParameters.size())
                self.assertEqual(
                        actualDimensionsOfParameters,
                        expectedDimensionsOfParameters)
                self.assertFalse(
                        torch.equal(newParameters, oldParameters))

    def _retrieveParameters(self, agent):
        agentParameters = []
        for layer in agent._layers:
            layerParameters = []
            copyOfWeights = layer.weight.data.clone().detach()
            layerParameters.append(copyOfWeights)
            copyOfBiases = layer.bias.data.clone().detach()
            layerParameters.append(copyOfBiases)
            agentParameters.append(layerParameters)
        return agentParameters
