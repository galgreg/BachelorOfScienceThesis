from src.training.TrainingResultsRepository import *
from src.training.TrainingLog import *
from src.AgentsPopulation import *
from src.AgentNeuralNetwork import *
import os
import os.path
from shutil import rmtree
import torch
import unittest
from unittest.mock import patch

# TODO -> dorobić testy dla metody self._repository.Save() + posprawdzać zabezpieczenia dla pozostałych metod
class TestTrainingResultsRepository(unittest.TestCase):
    def setUp(self):
        self._trainingLog = TrainingLog(isVerbose = False)
        self._repository = TrainingResultsRepository(self._trainingLog)
    
    def tearDown(self):
        del self._repository
        del self._trainingLog

    @patch('src.training.TrainingResultsRepository.datetime')
    def test_createLocationForTrainingResults(self, mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        basePath = os.path.dirname(os.path.realpath(__file__))
        basePathLastPart = "python_external_process"
        basePathEnd = \
                basePath.index(basePathLastPart) + len(basePathLastPart)        
        basePath = basePath[ : basePathEnd]
        self.assertTrue(basePath.endswith(basePathLastPart))
        basePath = os.path.join(basePath, "training_results")
        
        dirName = "1995_07_04_17_15_00"
        expectedLocation = os.path.join(basePath, dirName)
        actualLocation = self._repository._createLocationForTrainingResults()
        self.assertEqual(actualLocation, expectedLocation)
    
    def test_createBasePathForResults(self):
        expectedBasePath = os.path.dirname(os.path.realpath(__file__))
        basePathLastPart = "python_external_process"
        expectedBasePathEnd = \
                expectedBasePath.index(basePathLastPart) + len(basePathLastPart)        
        expectedBasePath = expectedBasePath[ : expectedBasePathEnd]
        self.assertTrue(expectedBasePath.endswith(basePathLastPart))
        expectedBasePath = os.path.join(expectedBasePath, "training_results")
        actualBasePath = self._repository._createBasePathForResults()
        self.assertEqual(actualBasePath, expectedBasePath)

    @patch('src.training.TrainingResultsRepository.datetime')
    def test_createDirNameForResults(self, mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        expectedDirName = "1995_07_04_17_15_00"
        actualDirName = self._repository._createDirNameForResults()
        self.assertEqual(actualDirName, expectedDirName)

    def test_saveBestModel_OK(self):
        population = \
                AgentsPopulation(
                        numberOfAgents = 5,
                        agentDimensions = [5, 3, 2],
                        learningAlgorithm = None)
        whichModelIsBest = 3
        
        locationForModel = "TEST_LOCATION_TO_SAVE_BEST_MODEL"
        if os.path.exists(locationForModel):
            rmtree(locationForModel)
        
        os.mkdir(locationForModel)
        self.assertTrue(os.path.isdir(locationForModel))
        
        self._repository._saveBestModel(
                locationForModel,
                population,
                whichModelIsBest)
        
        modelFilePath = os.path.join(locationForModel, "best_model.pth")
        doesModelFileExist = os.path.isfile(modelFilePath)
        self.assertTrue(doesModelFileExist)
        
        expectedSavedModel = population._agents[whichModelIsBest]
        actualSavedModel = torch.load(modelFilePath)
        self._compareModels(expectedSavedModel, actualSavedModel)
        
        rmtree(locationForModel)

    def test_saveWholePopulation_OK(self):
        population = \
                AgentsPopulation(
                        numberOfAgents = 5,
                        agentDimensions = [5, 3, 2],
                        learningAlgorithm = None)
        
        locationForPopulation = "TEST_LOCATION_TO_SAVE_BEST_POPULATION"
        if os.path.exists(locationForPopulation):
            rmtree(locationForPopulation)
        
        os.mkdir(locationForPopulation)
        self.assertTrue(os.path.isdir(locationForPopulation))
        
        self._repository._saveWholePopulation(locationForPopulation, population)
        pathToPopulationDir = os.path.join(locationForPopulation, "population")
        self.assertTrue(os.path.isdir(pathToPopulationDir))
        
        for i in range(len(population._agents)):
            fileNameForModel = "model_{0}.pth".format(i+1)
            fullPathToModelFile = \
                    os.path.join(pathToPopulationDir, fileNameForModel)
            self.assertTrue(os.path.isfile(fullPathToModelFile))
            
            expectedModel = population._agents[i]
            actualModel = torch.load(fullPathToModelFile)
            self._compareModels(actualModel, expectedModel)
        
        rmtree(locationForPopulation)

    def _compareModels(self, firstModel, secondModel):
        expectedTypeOfModel = AgentNeuralNetwork
        actualTypeOfModel_First = type(firstModel)
        self.assertEqual(actualTypeOfModel_First, expectedTypeOfModel)
        
        actualTypeOfModel_Second = type(secondModel)
        self.assertEqual(actualTypeOfModel_Second, expectedTypeOfModel)
        
        for firstLayer, secondLayer \
                in zip(firstModel._layers, secondModel._layers):
            expectedLayerType = torch.nn.Linear
            actualLayerType_First = type(firstLayer)
            actualLayerType_Second = type(secondLayer)
            
            self.assertEqual(firstLayer.in_features, secondLayer.in_features)
            self.assertEqual(firstLayer.out_features, secondLayer.out_features)
            
            firstLayerWeightParameters = firstLayer.weight.data
            secondLayerWeightParameters = secondLayer.weight.data
            self.assertTrue(
                    torch.equal(
                            firstLayerWeightParameters,
                            secondLayerWeightParameters))
            
            firstLayerBiasParameters = firstLayer.bias.data
            secondLayerBiasParameters = secondLayer.bias.data
            self.assertTrue(
                    torch.equal(
                            firstLayerBiasParameters,
                            secondLayerBiasParameters))
