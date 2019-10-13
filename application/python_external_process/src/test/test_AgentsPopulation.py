from src.AgentsPopulation import *
from ddt import ddt, data, unpack
import random
import torch
import torch.nn as nn
import unittest

@ddt
class TestAgentsPopulation(unittest.TestCase):
    @unpack
    @data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
    def test_Constructor_WithoutAlgorithm(
            self,
            expectedNumberOfAgents,
            agentDimensions):
        population = AgentsPopulation(
                expectedNumberOfAgents,
                agentDimensions,
                None)
        
        expectedSizeOfOutput = agentDimensions[-1]
        actualSizeOfOutput = population._sizeOfOutput
        self.assertEqual(actualSizeOfOutput, expectedSizeOfOutput)
        
        expectedTypeOfLearningAlgorithm = type(None)
        actualTypeOfLearningAlgorithm = type(population._learningAlgorithm)
        self.assertEqual(
                actualTypeOfLearningAlgorithm,
                expectedTypeOfLearningAlgorithm)

        actualNumberOfAgents = len(population._agents)
        self.assertEqual(actualNumberOfAgents, expectedNumberOfAgents)

    @unpack
    @data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
    def test_DoForward_AllAgentsAreNotDone(
            self,
            numberOfAgents,
            agentsDimensions):
        population = AgentsPopulation(numberOfAgents, agentsDimensions, None)
        
        listOfInputData = [
                [ random.uniform(0.0, 1.0) for j in range(agentsDimensions[0]) ]
                for i in range(numberOfAgents)
        ]
        agentDones = [ False ] * numberOfAgents
        
        listOfOutputData = population.DoForward(listOfInputData, agentDones)
        
        expectedTypeOfOutput = list
        actualTypeOfOutput = type(listOfOutputData)
        self.assertEqual(actualTypeOfOutput, expectedTypeOfOutput)
        
        expectedSizeOfOutput = numberOfAgents
        actualSizeOfOutput = len(listOfOutputData)
        self.assertEqual(actualSizeOfOutput, expectedSizeOfOutput)
        
        for outputData in listOfOutputData:
            expectedTypeOfData = list
            actualTypeOfData = type(outputData)
            self.assertEqual(actualTypeOfData, expectedTypeOfData)
            
            expectedSizeOfData = agentsDimensions[-1]
            actualSizeOfData = len(outputData)
            self.assertEqual(actualSizeOfData, expectedSizeOfData)
        
    @unpack
    @data((20, [2, 2], [1, 2, 15, 17]), (10, [5, 3, 2], [0, 3, 6, 9]))
    def test_DoForward_SomeAgentsAreDone(
            self,
            numberOfAgents,
            agentsDimensions,
            agentIndicesToDone):
        population = AgentsPopulation(numberOfAgents, agentsDimensions, None)
        
        listOfInputData = [
                [ random.uniform(0.0, 1.0) for j in range(agentsDimensions[0]) ]
                for i in range(numberOfAgents)
        ]
        agentDones = []
        for i in range(numberOfAgents):
            if i in agentIndicesToDone:
                agentDones.append(True)
            else:
                agentDones.append(False)
        
        listOfOutputData = population.DoForward(listOfInputData, agentDones)
        
        expectedTypeOfOutput = list
        actualTypeOfOutput = type(listOfOutputData)
        self.assertEqual(actualTypeOfOutput, expectedTypeOfOutput)
        
        expectedSizeOfOutput = numberOfAgents
        actualSizeOfOutput = len(listOfOutputData)
        self.assertEqual(actualSizeOfOutput, expectedSizeOfOutput)
        
        for i in range(numberOfAgents):
            if i in agentIndicesToDone:
                expectedOutputData = \
                        [0.0 for i in range(population._sizeOfOutput)]
                actualOutputData = listOfOutputData[i]
                self.assertEqual(actualOutputData, expectedOutputData)
            else:
                expectedTypeOfData = list
                actualTypeOfData = type(listOfOutputData[i])
                self.assertEqual(actualTypeOfData, expectedTypeOfData)
                
                expectedSizeOfData = agentsDimensions[-1]
                actualSizeOfData = len(listOfOutputData[i])
                self.assertEqual(actualSizeOfData, expectedSizeOfData)

    @unpack
    @data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
    def test_DoForward_AllAgentsAreDone(self, numberOfAgents, agentsDimensions):
        population = AgentsPopulation(numberOfAgents, agentsDimensions, None)
        listOfInputData = [
                [ random.uniform(0.0, 1.0) for j in range(agentsDimensions[0]) ]
                for i in range(numberOfAgents)
        ]
        agentDones = [ True ] * numberOfAgents
        
        listOfOutputData = population.DoForward(listOfInputData, agentDones)
        
        expectedTypeOfOutput = list
        actualTypeOfOutput = type(listOfOutputData)
        self.assertEqual(actualTypeOfOutput, expectedTypeOfOutput)
        
        expectedSizeOfOutput = numberOfAgents
        actualSizeOfOutput = len(listOfOutputData)
        self.assertEqual(actualSizeOfOutput, expectedSizeOfOutput)
        
        for actualOutputData in listOfOutputData:
            expectedOutputData = \
                    [0.0 for i in range(population._sizeOfOutput)]
            self.assertEqual(actualOutputData, expectedOutputData)
