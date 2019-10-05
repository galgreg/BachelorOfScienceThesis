from src.learning_algorithms.GeneticAlgorithm import *
from src.AgentNeuralNetwork import *
from ddt import ddt, data, unpack
import torch
import unittest

@ddt
class TestGeneticAlgorithm(unittest.TestCase):
	def setUp(self):
		self._algorithm = GeneticAlgorithm(
				selectionPercentRate = 10,
				probabilityThresholdToMutateChromosome = 0.8,
				probabilityThresholdToMutateGenome = 0.7)
		self._REAL_RANDRANGE = random.randrange
		self._REAL_SAMPLE = random.sample
	
	def tearDown(self):
		random.randrange = self._REAL_RANDRANGE
		random.sample = self._REAL_SAMPLE
		del self._algorithm
	
	@unpack
	@data((10, 0.8, 0.7), (1, 0.3, 0.9), (25, 0.9, 0.99))
	def test_InitialObjectState(
			self,
			expectedSelectionPercentRate,
			expectedProbabilityThresholdToMutateChromosome,
			expectedProbabilityThresholdToMutateGenome):
		tempAlgorithm = GeneticAlgorithm(
				expectedSelectionPercentRate,
				expectedProbabilityThresholdToMutateChromosome,
				expectedProbabilityThresholdToMutateGenome)
		actualSelectionPercentRate = tempAlgorithm._selectionPercentRate
		self.assertEqual(
				actualSelectionPercentRate,
				expectedSelectionPercentRate)
				
		actualProbabilityThresholdToMutateChromosome = \
				tempAlgorithm._probabilityThresholdToMutateChromosome
		self.assertEqual(
				actualProbabilityThresholdToMutateChromosome,
				expectedProbabilityThresholdToMutateChromosome)
				
		actualProbabilityThresholdToMutateGenome = \
				tempAlgorithm._probabilityThresholdToMutateGenome
		self.assertEqual(
				actualProbabilityThresholdToMutateGenome,
				expectedProbabilityThresholdToMutateGenome)
	@unpack
	@data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 7, 4, 2]))
	def test_ComputeNewPopulation(self, numberOfAgents, agentsDimensions):
		agentList = [
				AgentNeuralNetwork(agentsDimensions, False)
				for i in range(numberOfAgents)
		]
		fitnessList = [random.uniform(-1, 1) for i in range(numberOfAgents)]
		
		newPopulation = \
				self._algorithm.ComputeNewPopulation(agentList, fitnessList)
		
		expectedTypeOfNewPopulation = list
		actualTypeOfNewPopulation = type(newPopulation)
		self.assertEqual(actualTypeOfNewPopulation, expectedTypeOfNewPopulation)
		
		expectedSizeOfNewPopulation = numberOfAgents
		actualSizeOfNewPopulation = len(newPopulation)
		self.assertEqual(actualSizeOfNewPopulation, expectedSizeOfNewPopulation)
		
		for agent in agentList:
			expectedTypeOfAgent = AgentNeuralNetwork
			actualTypeOfAgent = type(agent)
			self.assertEqual(actualTypeOfAgent, expectedTypeOfAgent)
			
			for i in range(len(agentsDimensions) - 1):
				expectedInFeatureCount = agentsDimensions[i]
				actualInFeatureCount = agent._layers[i].in_features
				self.assertTrue(actualInFeatureCount, expectedInFeatureCount)
				
				expectedOutFeatureCount = agentsDimensions[i+1]
				actualOutFeatureCount = agent._layers[i].out_features
				self.assertTrue(actualOutFeatureCount, expectedOutFeatureCount)
	
	@unpack
	@data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
	def test_retrieveParametersFromAgents(self, numberOfAgents, agentsDimensions):
		agentList = [
				AgentNeuralNetwork(agentsDimensions, False)
				for i in range(numberOfAgents)
		]
		actualParameters = \
				self._algorithm._retrieveParametersFromAgents(agentList)
				
		expectedParametersType = torch.Tensor
		actualParametersType = type(actualParameters)
		self.assertEqual(actualParametersType, expectedParametersType)
		
		numberOfParameters = self._computeNumberOfParameters(agentsDimensions)
		
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
	
	@unpack
	@data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 7, 4, 2]))
	def test_setNewParametersOnAgentList(self, numberOfAgents, agentsDimensions):
		agentList = [
				AgentNeuralNetwork(agentsDimensions, False)
				for i in range(numberOfAgents)
		]
		numberOfParameters = self._computeNumberOfParameters(agentsDimensions)
		newParameters = torch.randn(numberOfAgents, numberOfParameters)
		
		parametersBeforeSet = self._retrieveParameters(agentList)
		newAgentList = \
				self._algorithm._setNewParametersOnAgentList(
						agentList,
						newParameters)
		parametersAfterSet = self._retrieveParameters(newAgentList)
		
		for oldAgentParameters, newAgentParameters \
				in zip(parametersBeforeSet, parametersAfterSet):
			for oldLayerParameters, newLayerParameters \
					in zip(oldAgentParameters, newAgentParameters):
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
	
	def _retrieveParameters(self, agentList):
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
	
	def _computeNumberOfParameters(self, agentsDimensions):
		numberOfParameters = 0
		numberOfLayers = len(agentsDimensions) - 1
		for i in range(numberOfLayers):
			numberOfWeights = agentsDimensions[i] * agentsDimensions[i+1]
			numberOfBiases = agentsDimensions[i+1]
			parametersPerLayer = numberOfWeights + numberOfBiases
			numberOfParameters = numberOfParameters + parametersPerLayer
		return numberOfParameters
	
	def test_doSelection(self):
		randRangeContent = [3, 0, 1, 4, 3, 2, 3, 5]
		random.randrange = lambda n: randRangeContent.pop(0)
		
		oldPopulation = torch.randn(8, 10)
		fitnessList = [-10, 15.99, -21.13, 124.11, -3.14, 17, 22.3, 5.44]
		self._algorithm._selectionPercentRate = 25
		actualParentPool = \
				self._algorithm._doSelection(oldPopulation, fitnessList)
		
		expectedSizeOfParentPool = 2
		actualSizeOfParentPool = len(actualParentPool)
		self.assertEqual(actualSizeOfParentPool, expectedSizeOfParentPool)
		
		expectedParentPool = [ oldPopulation[1], oldPopulation[3] ]
		for i in range(actualSizeOfParentPool):
			self.assertTrue(
					torch.equal(actualParentPool[i], expectedParentPool[i])
			)

	def test_doCrossover(self):
		parentPool = [
				torch.tensor([1, 2, 3, 4, 5, 6]),
				torch.tensor([10, 20, 30, 40, 50, 60]),
				torch.tensor([11, 22, 33, 44, 55, 66]),
				torch.tensor([100, 200, 300, 400, 500, 600]),
				torch.tensor([123, 234, 345, 456, 567, 678]),
				torch.tensor([14, 25, 36, 47, 58, 69])
		]
		random.sample = lambda a, b: [parentPool[i] for i in [0, 5, 3, 2, 4, 1]]
		crossoverPoint = 3
		random.randrange = lambda n : crossoverPoint
		
		selectionSize = len(parentPool)
		cloningCount = 10
		sizeOfNewPopulation = selectionSize * cloningCount
		sizeOfChromosome = 6
		
		childrenPrefabs = [
				torch.tensor([1, 2, 3, 47, 58, 69]),
				torch.tensor([14, 25, 36, 4, 5, 6]),
				torch.tensor([100, 200, 300, 44, 55, 66]),
				torch.tensor([11, 22, 33, 400, 500, 600]),
				torch.tensor([123, 234, 345, 40, 50, 60]),
				torch.tensor([10, 20, 30, 456, 567, 678])
		]
		expectedNewPopulation = torch.stack(childrenPrefabs * cloningCount)
		actualNewPopulation = \
				self._algorithm._doCrossover(
						parentPool,
						sizeOfNewPopulation,
						sizeOfChromosome)
		
		self.assertEqual(len(actualNewPopulation), len(expectedNewPopulation))
		for actualChromosome, expectedChromosome in zip(actualNewPopulation, expectedNewPopulation):
			self.assertTrue(torch.equal(actualChromosome, expectedChromosome))
	
	@unpack
	@data((8, [0, 5, 3, 2, 6, 1, 4, 7]), (5, [3, 2, 4, 0, 1]))
	def test_createParentPairs(self, sizeOfParentPool, sampleIndices):
		parentPool = [torch.randn(10) ] * sizeOfParentPool
		sampledParentPool = [ parentPool[i] for i in sampleIndices ]
		random.sample = lambda a, b: sampledParentPool
		from math import ceil
		expectedParentPairs = [
				sampledParentPool[2*i : 2*i + 2]
				for i in range(ceil(len(sampledParentPool) / 2))
		]
		actualParentPairs = self._algorithm._createParentPairs(parentPool)
		
		self.assertEqual(len(actualParentPairs), len(expectedParentPairs))
		for actualPair, expectedPair in zip(actualParentPairs, expectedParentPairs):
			self.assertEqual(len(actualPair), len(expectedPair))
			for actualParent, expectedParent in zip(actualPair, expectedPair):
				self.assertTrue(torch.equal(actualParent, expectedParent))

	def test_pickCrossoverPoint(self):
		expectedCrossoverPoint = 13
		random.randrange = lambda n: expectedCrossoverPoint
		sizeOfChromosome = 20
		actualCrossoverPoint = \
				self._algorithm._pickCrossoverPoint(sizeOfChromosome)
		self.assertEqual(actualCrossoverPoint, expectedCrossoverPoint)

	def test_createChildrenPrefabs_ChildrenHaveTwoParents(self):
		parentsPair = [
				torch.tensor([1, 2, 3, 4, 5, 6]),
				torch.tensor([100, 200, 300, 400, 500, 600])
		]
		crossoverPoint = 3
		expectedPrefabs = [
				torch.tensor([1, 2, 3, 400, 500, 600]),
				torch.tensor([100, 200, 300, 4, 5, 6])
		]
		actualPrefabs = self._algorithm._createChildrenPrefabs(
				parentsPair,
				crossoverPoint)
		self.assertEqual(len(actualPrefabs), len(expectedPrefabs))
		self.assertTrue(torch.equal(actualPrefabs[0], expectedPrefabs[0]))
		self.assertTrue(torch.equal(actualPrefabs[1], expectedPrefabs[1]))

	def test_createChildrenPrefabs_ChildrenHaveOnlyOneParent(self):
		singleParent = [ torch.tensor([1, 2, 3, 4, 5, 6]) ]
		crossoverPoint = 3
		expectedPrefabs = singleParent * 2
		actualPrefabs = self._algorithm._createChildrenPrefabs(
				singleParent,
				crossoverPoint)
		self.assertEqual(len(actualPrefabs), len(expectedPrefabs))
		self.assertTrue(torch.equal(actualPrefabs[0], expectedPrefabs[0]))
		self.assertTrue(torch.equal(actualPrefabs[1], expectedPrefabs[1]))

	@data(0, 3, 10, 20)
	def test_createChildrenPrefabs_InvalidNumberOfParents(self, numberOfParents):
		parents = [ torch.randn(10) ] * numberOfParents
		crossoverPoint = 3
		self.assertRaises(
				ValueError,
				self._algorithm._createChildrenPrefabs,
				parents,
				crossoverPoint)
	@unpack
	@data((10, 100), (13, 100), (7, 50))
	def test_createNewPopulationFromPrefabs(
			self,
			numberOfPrefabs,
			sizeOfNewPopulation):
		childrenPrefabs = [ torch.randn(10) ] * numberOfPrefabs
		
		from math import ceil
		expectedPopulation = \
				childrenPrefabs * ceil(sizeOfNewPopulation / numberOfPrefabs)
		expectedPopulation = expectedPopulation[0 : sizeOfNewPopulation]
		self.assertEqual(len(expectedPopulation), sizeOfNewPopulation)
		
		actualPopulation = self._algorithm._createNewPopulationFromPrefabs(
				childrenPrefabs,
				sizeOfNewPopulation)
		self.assertEqual(len(actualPopulation), len(expectedPopulation))
		
		for actualChromosome, expectedChromosome in zip(actualPopulation, expectedPopulation):
			self.assertTrue(torch.equal(actualChromosome, expectedChromosome))

	@unpack
	@data((-1.0, -1.0, False), (-1.0, 2.0, True), (2.0, 2.0, True))
	def test_doMutation(
			self,
			probabilityThresholdToMutateChromosome,
			probabilityThresholdToMutateGenome,
			expectedArePopulationsEqual):
		newPopulationToMutate = torch.randn(10, 10)
		populationBeforeMutation = newPopulationToMutate.clone().detach()
		
		self._algorithm._probabilityThresholdToMutateChromosome = \
				probabilityThresholdToMutateChromosome
		self._algorithm._probabilityThresholdToMutateGenome = \
				probabilityThresholdToMutateGenome
		
		self._algorithm._doMutation(newPopulationToMutate)
		
		populationAfterMutation = newPopulationToMutate.clone().detach()
		actualArePopulationsEqual = \
				torch.equal(populationAfterMutation, populationBeforeMutation)
		self.assertEqual(actualArePopulationsEqual, expectedArePopulationsEqual)

	@unpack
	@data((-1.0, False), (2.0, True))
	def test_mutateChromosome(
			self,
			probabilityThresholdToMutateGenome,
			expectedAreChromosomesEqual):
		chromosomeToMutate = torch.randn(10)
		chromosomeBeforeMutation = chromosomeToMutate.clone().detach()
		
		self._algorithm._probabilityThresholdToMutateGenome = \
				probabilityThresholdToMutateGenome
		self._algorithm._mutateChromosome(chromosomeToMutate)
		
		chromosomeAfterMutation = chromosomeToMutate.clone().detach()

		actualAreChromosomesEqual = \
				torch.equal(chromosomeAfterMutation, chromosomeBeforeMutation)
		self.assertEqual(actualAreChromosomesEqual, expectedAreChromosomesEqual)
