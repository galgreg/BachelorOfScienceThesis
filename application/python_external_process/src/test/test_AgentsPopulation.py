from src.AgentsPopulation import *
from src.learning_algorithms.GeneticAlgorithm import *
from ddt import ddt, data, unpack
import random
import torch
import torch.nn as nn
import unittest

@ddt
class TestAgentsPopulation(unittest.TestCase):
	@unpack
	@data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
	def test_Constructor_WithGeneticAlgorithm(
			self,
			expectedNumberOfAgents,
			agentDimensions):
		population = AgentsPopulation(
				expectedNumberOfAgents,
				agentDimensions,
				GeneticAlgorithm(10, 0.8, 0.7))
		
		expectedTypeOfLearningAlgorithm = GeneticAlgorithm
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
	@data((20, [2, 1], [1, 2, 15, 17]), (10, [5, 3, 2], [0, 3, 6, 9]))
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
				self.assertTrue(listOfOutputData[i] is None)
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
		
		for outputData in listOfOutputData:
			self.assertTrue(outputData is None)

	@unpack
	@data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
	def test_Learn_GeneticAlgorithm(self, numberOfAgents, agentDimensions):
		population = AgentsPopulation(
				numberOfAgents,
				agentDimensions,
				GeneticAlgorithm(
					selectionPercentRate = 10,
					probabilityThresholdToMutateChromosome = 0.0,
					probabilityThresholdToMutateGenome = 0.0))
		
		rewardList = [random.uniform(-1, 1) for i in range(numberOfAgents)]
		
		agentsParametersBeforeLearn = \
				self._retrieveLearnableParameters(population._agents)
		population.Learn(rewardList)
		agentsParametersAfterLearn = \
				self._retrieveLearnableParameters(population._agents)
		
		expectedTypeOfAgents = list
		actualTypeOfAgents = type(population._agents)
		self.assertEqual(actualTypeOfAgents, expectedTypeOfAgents)
		
		expectedSizeOfAgentList = numberOfAgents
		actualSizeOfAgentList = len(population._agents)
		self.assertEqual(actualSizeOfAgentList, expectedSizeOfAgentList)
		
		for agent in population._agents:
			expectedTypeOfAgent = AgentNeuralNetwork
			actualTypeOfAgent = type(agent)
			self.assertEqual(actualTypeOfAgent, expectedTypeOfAgent)
			
			expectedTypeOfLayers = list
			actualTypeOfLayers = type(agent._layers)
			self.assertEqual(actualTypeOfLayers, expectedTypeOfLayers)
			
			expectedSizeOfLayers = len(agentDimensions) - 1
			actualSizeOfLayers = len(agent._layers)
			self.assertEqual(actualSizeOfLayers, expectedSizeOfLayers)
			
			for i in range(len(agentDimensions) - 1):
				expectedTypeOfLayer = nn.Linear
				actualTypeOfLayer = type(agent._layers[i])
				self.assertEqual(actualTypeOfLayer, expectedTypeOfLayer)
				
				expectedInFeaturesCount = agentDimensions[i]
				actualInFeaturesCount = agent._layers[i].in_features
				self.assertEqual(actualInFeaturesCount, expectedInFeaturesCount)
				
				expectedOutFeaturesCount = agentDimensions[i+1]
				actualOutFeaturesCount = agent._layers[i].out_features
				self.assertEqual(actualOutFeaturesCount, expectedOutFeaturesCount)
				
		for oldAgentParameters, newAgentParameters in \
				zip(agentsParametersBeforeLearn, agentsParametersAfterLearn):
			for oldLayerParameters, newLayerParameters in \
					zip(oldAgentParameters, newAgentParameters):
				oldWeightSize = list(oldLayerParameters[0].size())
				newWeightSize = list(newLayerParameters[0].size())
				self.assertEqual(newWeightSize, oldWeightSize)
				
				oldBiasSize = list(oldLayerParameters[1].size())
				newBiasSize = list(newLayerParameters[1].size())
				self.assertEqual(newBiasSize, oldBiasSize)
						
				self.assertFalse(
						torch.equal(
								oldLayerParameters[0],
								newLayerParameters[0]))
				self.assertFalse(
						torch.equal(
								oldLayerParameters[1],
								newLayerParameters[1]))

	
	def _retrieveLearnableParameters(self, agentList):
		populationParameters = []
		for agent in agentList:
			agentParameters = []
			for layer in agent._layers:
				layerParameters = []
				copyOfWeights = layer.weight.data.clone().detach()
				layerParameters.append(copyOfWeights)
				copyOfBiases = layer.bias.data.clone().detach()
				layerParameters.append(copyOfBiases)
				agentParameters.append(layerParameters)
			populationParameters.append(agentParameters)
		
		return populationParameters
