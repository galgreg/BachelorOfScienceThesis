from src.genetic.GeneticAlgorithm import *
from ddt import ddt, data, unpack
import torch
import unittest

@ddt
class TestGeneticAlgorithm(unittest.TestCase):
	def setUp(self):
		self._algorithm = GeneticAlgorithm(selectionPercentRate = 10)
	
	def tearDown(self):
		del self._algorithm
	
	def validatePopulationContent(self, population):
		for chromosome in population:
			for genome in chromosome:
				self.assertTrue(genome >= 0.0)
				self.assertTrue(genome <= 1.0)
	
	def test_StateOfObjectAtTheBeginning_CreatedByDefaultConstructor(self):
		expectedSelectionRate = 10
		actualSelectionRate = self._algorithm._selectionPercentRate
		self.assertEqual(actualSelectionRate, expectedSelectionRate)
		
	def test_StateOfObjectAtTheBeginning_SpecifySelectionRateWhileCreating(self):
		expectedSelectionRate = 25
		tempAlgorithmObject = GeneticAlgorithm(expectedSelectionRate)
		actualSelectionRate = tempAlgorithmObject._selectionPercentRate
		self.assertEqual(actualSelectionRate, expectedSelectionRate)
		
	def test_InitPopulation(self):
		populationSize = 10
		chromosomeSize = 20
		self._algorithm.InitPopulation(populationSize, chromosomeSize)
		
		expectedPopulationSize = populationSize
		actualPopulationSize = self._algorithm._populationSize
		self.assertEqual(actualPopulationSize, expectedPopulationSize)
		
		expectedPopulationDimensions = \
				torch.Size([populationSize, chromosomeSize])
		actualPopulationDimensions = self._algorithm._population.size()
		self.assertEqual(
				actualPopulationDimensions,
				expectedPopulationDimensions)
				
		self.validatePopulationContent(self._algorithm._population)
	
	def test_GetPopulation(self):
		self._algorithm._population = torch.ones(10, 5)
		expectedPopulation = self._algorithm._population
		actualPopulation = self._algorithm.GetPopulation()
		self.assertTrue(torch.equal(actualPopulation, expectedPopulation))
	
	def test_DoTournamentSelection(self):
		self._algorithm._selectionPercentRate = 25
		self._algorithm._population = torch.randn(8, 10)
		randRangeContent = [3, 0, 1, 4, 3, 2, 3, 5]
		random.randrange = lambda n: randRangeContent.pop(0)
		
		fitnessList = [-10, 15.99, -21.13, 124.11, -3.14, 17, 22.3, 5.44]
		actualParentPool = self._algorithm.DoTournamentSelection(fitnessList)
		
		expectedSizeOfParentPool = 2
		actualSizeOfParentPool = len(actualParentPool)
		self.assertEqual(actualSizeOfParentPool, expectedSizeOfParentPool)
		
		expectedParentPool = [
				self._algorithm._population[1],
				self._algorithm._population[3]
		]
		for i in range(actualSizeOfParentPool):
			self.assertTrue(
					torch.equal(actualParentPool[i], expectedParentPool[i])
			)
	
	@unpack
	@data((100, 5), (2, 1), (1, 1))
	def test_computeContendersCount(self, sizeOfPopulation, expectedContendersCount):
		self._algorithm._population = list(range(sizeOfPopulation))
		random.randrange = lambda n: int(n / 2)
		actualContendersCount = self._algorithm._computeContendersCount()
		self.assertEqual(actualContendersCount, expectedContendersCount)	
	
	@unpack
	@data((100, 10), (20, 2), (10, 1), (5, 1), (1, 1))
	def test_computeSizeOfParentPool(self, sizeOfPopulation, expectedSizeOfParentPool):
		self._algorithm._population = list(range(sizeOfPopulation))
		actualSizeOfParentPool = self._algorithm._computeSizeOfParentPool()
		self.assertEqual(actualSizeOfParentPool, expectedSizeOfParentPool)
	
	def test_getChromosomesToCompete(self):
		self._algorithm._population = torch.randn(5, 10)
		fitnessList = [10.51, -20.3, -30.99, 40.11, 50.0]
		indicesToChoose = [0, 1, 3]
		random.randrange = lambda n : indicesToChoose.pop(0)
		expectedChromosomes = { \
				fitnessList[0] : self._algorithm._population[0], \
				fitnessList[1] : self._algorithm._population[1], \
				fitnessList[3] : self._algorithm._population[3] \
		}
		actualChromosomes = \
				self._algorithm._getChromosomesToCompete(
						fitnessList,
						len(indicesToChoose))
						
		self.assertEqual(actualChromosomes.keys(), expectedChromosomes.keys())
		for actualKey in actualChromosomes.keys():
			self.assertTrue(
					torch.equal(
							actualChromosomes[actualKey],
							expectedChromosomes[actualKey]))
	
	def test_getBestChromosome(self):
		expectedBestChromosome = torch.randn(10)
		chromosomesToChoose = {\
				10.51 : torch.randn(10), \
				-20.3 : torch.randn(10), \
				40.11 : expectedBestChromosome \
		}
		actualBestChromosome = \
				self._algorithm._getBestChromosome(chromosomesToChoose)
		self.assertTrue(torch.equal(actualBestChromosome, expectedBestChromosome))

	def test_DoOnePointCrossover(self):
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
		cloningCount = 1
		self._algorithm._populationSize = selectionSize * cloningCount
		self._algorithm._chromosomeSize = 6
		
		childrenPrefabs = [
				torch.tensor([1, 2, 3, 47, 58, 69]),
				torch.tensor([14, 25, 36, 4, 5, 6]),
				torch.tensor([100, 200, 300, 44, 55, 66]),
				torch.tensor([11, 22, 33, 400, 500, 600]),
				torch.tensor([123, 234, 345, 40, 50, 60]),
				torch.tensor([10, 20, 30, 456, 567, 678])
		]
		expectedNewPopulation = torch.stack(childrenPrefabs * cloningCount)
		self._algorithm.DoOnePointCrossover(parentPool)
		actualNewPopulation = self._algorithm._population
		
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
	def test_createNewPopulationFromPrefabs(self, numberOfPrefabs, populationSize):
		childrenPrefabs = [ torch.randn(10) ] * numberOfPrefabs
		self._algorithm._populationSize = populationSize
		
		from math import ceil
		expectedPopulation = childrenPrefabs * ceil(populationSize / numberOfPrefabs)
		expectedPopulation = expectedPopulation[0 : populationSize]
		self.assertEqual(len(expectedPopulation), populationSize)
		
		actualPopulation = self._algorithm._createNewPopulationFromPrefabs(
				childrenPrefabs)
		self.assertEqual(len(actualPopulation), len(expectedPopulation))
		
		for actualChromosome, expectedChromosome in zip(actualPopulation, expectedPopulation):
			self.assertTrue(torch.equal(actualChromosome, expectedChromosome))

	@unpack
	@data((-1.0, -1.0, False), (-1.0, 2.0, True), (2.0, 2.0, True))
	def test_DoMutation(
			self,
			probabilityThresholdToMutateChromosome,
			probabilityThresholdToMutateGenome,
			expectedArePopulationsEqual):
		self._algorithm._population = torch.randn(10, 10)
		populationBeforeMutation = self._algorithm._population.clone().detach()
		
		self._algorithm.DoMutation(
				probabilityThresholdToMutateChromosome,
				probabilityThresholdToMutateGenome)
		
		populationAfterMutation = self._algorithm._population.clone().detach()
		actualArePopulationsEqual = \
				torch.equal(populationAfterMutation, populationBeforeMutation)

	@unpack
	@data((-1.0, False), (2.0, True))
	def test_mutateChromosome(
			self,
			probabilityThresholdToMutateGenome,
			expectedAreChromosomesEqual):
		chromosomeToMutate = torch.randn(10)
		chromosomeBeforeMutation = chromosomeToMutate.clone().detach()
		
		self._algorithm._mutateChromosome(
				chromosomeToMutate,
				probabilityThresholdToMutateGenome)
		chromosomeAfterMutation = chromosomeToMutate.clone().detach()

		actualAreChromosomesEqual = \
				torch.equal(chromosomeAfterMutation, chromosomeBeforeMutation)
		self.assertEqual(actualAreChromosomesEqual, expectedAreChromosomesEqual)
