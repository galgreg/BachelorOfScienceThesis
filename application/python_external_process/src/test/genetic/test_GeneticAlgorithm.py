from src.genetic.GeneticAlgorithm import *
from ddt import ddt, data, unpack
import torch
import unittest

@ddt
class TestGeneticAlgorithm(unittest.TestCase):
	def setUp(self):
		self._algorithm = GeneticAlgorithm()
	
	def tearDown(self):
		del self._algorithm
	
	def validatePopulationContent(self, population):
		for chromosome in population:
			for genome in chromosome:
				self.assertTrue(genome > 0.0)
				self.assertTrue(genome < 1.0)
	
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
		expectedPopulationDimensions = \
				torch.Size([populationSize, chromosomeSize])
		initialPopulation = self._algorithm._population
		actualPopulationDimensions = initialPopulation.size()
		self.assertEqual(
				actualPopulationDimensions,
				expectedPopulationDimensions)
		self.validatePopulationContent(initialPopulation)
	
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
	
	@data((100, 5), (2, 1), (1, 1))
	@unpack
	def test_computeContendersCount(self, sizeOfPopulation, expectedContendersCount):
		self._algorithm._population = list(range(sizeOfPopulation))
		random.randrange = lambda n: int(n / 2)
		actualContendersCount = self._algorithm._computeContendersCount()
		self.assertEqual(actualContendersCount, expectedContendersCount)	
	
	@data((100, 10), (20, 2), (10, 1), (5, 1), (1, 1))
	@unpack
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
