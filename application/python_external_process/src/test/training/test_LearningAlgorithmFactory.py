from src.training.LearningAlgorithmFactory import *
from src.training.TrainingLog import *
import unittest

class TestLearningAlgorithmFactory(unittest.TestCase):
    def setUp(self):
        self._options = {"--genetic": True, "--neat" : False, "--ppo" : False}
        self.CONFIG_DATA = {
            "LearningAlgorithms" : {
                "ppo" : { },
                "neat" : { },
                "genetic" : {
                    "selectionPercentRate" : 10,
                    "minimalProbabilityToMutateChromosome" : 0.8,
                    "minimalProbabilityToMutateGenome" : 0.7
                }
            }
        }
        self._trainingLog = TrainingLog(isVerbose = False)
    
    def tearDown(self):
        del self._trainingLog
        del self.CONFIG_DATA
        del self._options
    
    def test_Constructor_WithoutTrainingLog(self):
        factory = LearningAlgorithmFactory(self._options, self.CONFIG_DATA)
        
        expectedOptions = self._options
        actualOptions = factory._options
        self.assertEqual(actualOptions, expectedOptions)
        
        expectedConfigData = self.CONFIG_DATA
        actualConfigData = factory._configData
        self.assertEqual(actualConfigData, expectedConfigData)
        
        self.assertTrue(factory._trainingLog is None)

    def test_Constructor_WithTrainingLog(self):
        factory = LearningAlgorithmFactory(
                self._options,
                self.CONFIG_DATA,
                self._trainingLog)
        
        expectedOptions = self._options
        actualOptions = factory._options
        self.assertEqual(actualOptions, expectedOptions)
        
        expectedConfigData = self.CONFIG_DATA
        actualConfigData = factory._configData
        self.assertEqual(actualConfigData, expectedConfigData)
        
        expectedTrainingLogType = TrainingLog
        actualTrainingLogType = type(factory._trainingLog)
        self.assertEqual(actualTrainingLogType, expectedTrainingLogType)

    def test_Create_GeneticAlgorithm(self):
        factory = LearningAlgorithmFactory(
                self._options,
                self.CONFIG_DATA,
                self._trainingLog)
        algorithmParameters = self.CONFIG_DATA["LearningAlgorithms"]["genetic"]
        
        createdObject = factory.Create()
        
        expectedTypeOfObject = GeneticAlgorithm
        actualTypeOfObject = type(createdObject)
        self.assertEqual(actualTypeOfObject, expectedTypeOfObject)
        
        expectedSelectionRate = \
                algorithmParameters["selectionPercentRate"]
        actualSelectionRate = \
                createdObject._selectionPercentRate
        self.assertEqual(actualSelectionRate, expectedSelectionRate)
        
        expectedProbabilityThresholdToMutateChromosome = \
                algorithmParameters["minimalProbabilityToMutateChromosome"]
        actualProbabilityThresholdToMutateChromosome = \
                createdObject._probabilityThresholdToMutateChromosome
        self.assertEqual(
                actualProbabilityThresholdToMutateChromosome,
                expectedProbabilityThresholdToMutateChromosome)
        
        expectedProbabilityThresholdToMutateGenome = \
                algorithmParameters["minimalProbabilityToMutateGenome"]
        actualProbabilityThresholdToMutateGenome = \
                createdObject._probabilityThresholdToMutateGenome
        self.assertEqual(
                actualProbabilityThresholdToMutateGenome,
                expectedProbabilityThresholdToMutateGenome)
        
        expectedLogMessage = \
                "Created GeneticAlgorithm with parameters: " \
                "selectionPercentRate = {0}, " \
                "probabilityThresholdToMutateChromosome = {1}, " \
                "probabilityThresholdToMutateGenome = {2}\n".format(
                        expectedSelectionRate,
                        expectedProbabilityThresholdToMutateChromosome,
                        expectedProbabilityThresholdToMutateGenome)
        actualLogMessage = factory._trainingLog._content
        self.assertEqual(actualLogMessage, expectedLogMessage)
                

    def test_Create_Neat(self):
        self._options = {"--genetic": False, "--neat" : True, "--ppo" : False}
        factory = LearningAlgorithmFactory(self._options, self.CONFIG_DATA)
        self.assertRaises(NotImplementedError, factory.Create)
    
    def test_Create_PPO(self):
        self._options = {"--genetic": False, "--neat" : False, "--ppo" : True}
        factory = LearningAlgorithmFactory(self._options, self.CONFIG_DATA)
        self.assertRaises(NotImplementedError, factory.Create)

    def test_createGeneticAlgorithm(self):
        factory = LearningAlgorithmFactory(
                self._options,
                self.CONFIG_DATA,
                self._trainingLog)
        algorithmParameters = self.CONFIG_DATA["LearningAlgorithms"]["genetic"]
        
        createdObject = factory._createGeneticAlgorithm()
        
        expectedTypeOfObject = GeneticAlgorithm
        actualTypeOfObject = type(createdObject)
        self.assertEqual(actualTypeOfObject, expectedTypeOfObject)
        
        expectedSelectionRate = \
                algorithmParameters["selectionPercentRate"]
        actualSelectionRate = \
                createdObject._selectionPercentRate
        self.assertEqual(actualSelectionRate, expectedSelectionRate)
        
        expectedProbabilityThresholdToMutateChromosome = \
                algorithmParameters["minimalProbabilityToMutateChromosome"]
        actualProbabilityThresholdToMutateChromosome = \
                createdObject._probabilityThresholdToMutateChromosome
        self.assertEqual(
                actualProbabilityThresholdToMutateChromosome,
                expectedProbabilityThresholdToMutateChromosome)
        
        expectedProbabilityThresholdToMutateGenome = \
                algorithmParameters["minimalProbabilityToMutateGenome"]
        actualProbabilityThresholdToMutateGenome = \
                createdObject._probabilityThresholdToMutateGenome
        self.assertEqual(
                actualProbabilityThresholdToMutateGenome,
                expectedProbabilityThresholdToMutateGenome)
        
        expectedLogMessage = \
                "Created GeneticAlgorithm with parameters: " \
                "selectionPercentRate = {0}, " \
                "probabilityThresholdToMutateChromosome = {1}, " \
                "probabilityThresholdToMutateGenome = {2}\n".format(
                        expectedSelectionRate,
                        expectedProbabilityThresholdToMutateChromosome,
                        expectedProbabilityThresholdToMutateGenome)
        actualLogMessage = factory._trainingLog._content
        self.assertEqual(actualLogMessage, expectedLogMessage)
