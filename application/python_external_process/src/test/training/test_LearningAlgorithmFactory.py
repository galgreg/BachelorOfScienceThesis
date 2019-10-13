from src.training.LearningAlgorithmFactory import *
from src.training.TrainingLog import *
import unittest
from unittest.mock import patch

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

    def test_Create_Neat(self):
        self._options = {"--genetic": False, "--neat" : True, "--ppo" : False}
        factory = LearningAlgorithmFactory(self._options, self.CONFIG_DATA)
        self.assertRaises(NotImplementedError, factory.Create)
    
    def test_Create_PPO(self):
        self._options = {"--genetic": False, "--neat" : False, "--ppo" : True}
        factory = LearningAlgorithmFactory(self._options, self.CONFIG_DATA)
        self.assertRaises(NotImplementedError, factory.Create)
