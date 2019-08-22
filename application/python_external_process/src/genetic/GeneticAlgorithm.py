import random
import torch

class GeneticAlgorithm:
	def __init__(self, selectionPercentRate=10):
		self._selectionPercentRate = selectionPercentRate
	
	def InitPopulation(self, populationSize, chromosomeSize):
		self._population = torch.FloatTensor(populationSize, chromosomeSize)
		self._population = self._population.uniform_(0.0, 1.0)
	
	def GetPopulation(self):
		return self._population
		
	def DoTournamentSelection(self, fitnessScoreList):
		parentPool = []
		sizeOfParentPool = self._computeSizeOfParentPool()
		contendersCount = self._computeContendersCount()
		for i in range(sizeOfParentPool):
			chromosomesToCompete = \
					self._getChromosomesToCompete(
							fitnessScoreList,
							contendersCount)
			bestChromosome = self._getBestChromosome(chromosomesToCompete)
			parentPool.append(bestChromosome)
		return parentPool
	
	def _computeSizeOfParentPool(self):
		sizeOfParentPool = \
				int(len(self._population) * self._selectionPercentRate / 100)
		if sizeOfParentPool < 1:
			sizeOfParentPool = 1
		return sizeOfParentPool
		
	def _computeContendersCount(self):
		from math import sqrt
		maxContendersCount = int(sqrt(len(self._population)))
		contendersCount = random.randrange(maxContendersCount)
		if contendersCount < 1:
			contendersCount = 1
		return contendersCount

	def _getChromosomesToCompete(self, fitnessScoreList, contendersCount):
		fitnessToChromosomesDict = dict(zip(fitnessScoreList, self._population))
		chromosomesToChoose = {}
		for i in range(0, contendersCount):
			fitnessIndex = random.randrange(len(fitnessScoreList))
			chosenFitness = fitnessScoreList[fitnessIndex]
			chosenChromosome = fitnessToChromosomesDict[chosenFitness]
			chromosomesToChoose[chosenFitness] = chosenChromosome			
		return chromosomesToChoose
		
	def _getBestChromosome(self, chromosomesToChoose):
		chromosomeFitnesses = chromosomesToChoose.keys()
		bestFitness = max(chromosomeFitnesses)
		return chromosomesToChoose[bestFitness]
