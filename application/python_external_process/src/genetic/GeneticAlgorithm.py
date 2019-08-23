import random
import torch

class GeneticAlgorithm:
	def __init__(self, selectionPercentRate = 10):
		self._selectionPercentRate = selectionPercentRate

	def InitPopulation(self, populationSize, chromosomeSize):
		self._populationSize = populationSize
		self._chromosomeSize = chromosomeSize
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

	def DoOnePointCrossover(self, parentPool):
		parentPairs = self._createParentPairs(parentPool)
		childrenPrefabsPool = []
		for i in range(len(parentPairs)):
			crossoverPoint = self._pickCrossoverPoint(self._chromosomeSize)
			firstChild, secondChild = \
					self._createChildrenPrefabs(parentPairs[i], crossoverPoint)
			childrenPrefabsPool.append(firstChild)
			childrenPrefabsPool.append(secondChild)
		newPopulation = self._createNewPopulationFromPrefabs(childrenPrefabsPool)
		self._population = torch.stack(newPopulation)

	def DoMutation(
			self,
			probabilityThresholdToMutateChromosome,
			probabilityThresholdToMutateGenome):
		for i in range(len(self._population)):
			tempRandom = random.random()
			if tempRandom >= probabilityThresholdToMutateChromosome:
				self._mutateChromosome(
						self._population[i],
						probabilityThresholdToMutateGenome)

	
	def _createParentPairs(self, parentPool):
		sampledParentPool = random.sample(parentPool, len(parentPool))
		sizeOfPair = 2
		parentPairs = [
				sampledParentPool[i : i + sizeOfPair]
				for i in range(0, len(sampledParentPool), sizeOfPair)
		]
		return parentPairs
	
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
	
	def _pickCrossoverPoint(self, sizeOfChromosome):
		crossoverPoint = random.randrange(sizeOfChromosome)
		return crossoverPoint

	def _createChildrenPrefabs(self, parents, crossOverPoint):
		if len(parents) == 2:
			firstChild = torch.cat((
					parents[0][ : crossOverPoint],
					parents[1][crossOverPoint : ]))
			secondChild = torch.cat((
					parents[1][ : crossOverPoint],
					parents[0][crossOverPoint : ]))
			return [firstChild, secondChild]
		elif len(parents) == 1:
			return parents * 2
		else:
			raise ValueError("GeneticAlgorithm._createChildrenPrefabs error: "
					"parents has invalid size! (should be 1 or 2)!")

	def _createNewPopulationFromPrefabs(self, childrenPrefabs):
		from math import ceil
		cloneCounter = ceil(self._populationSize / len(childrenPrefabs))
		newPopulation = childrenPrefabs * cloneCounter
		newPopulation = newPopulation[0 : self._populationSize]
		return newPopulation
	
	def _mutateChromosome(self, chromosome, probabilityThresholdToMutateGenome):
		for i in range(len(chromosome)):
			tempRandom = random.random()
			if tempRandom >= probabilityThresholdToMutateGenome:
				chromosome[i] = tempRandom
