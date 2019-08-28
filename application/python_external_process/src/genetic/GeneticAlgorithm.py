import random
import torch

class GeneticAlgorithm:
	def __init__(
			self,
			selectionPercentRate,
			probabilityThresholdToMutateChromosome,
			probabilityThresholdToMutateGenome):
		self._selectionPercentRate = selectionPercentRate
		self._probabilityThresholdToMutateChromosome = \
				probabilityThresholdToMutateChromosome
		self._probabilityThresholdToMutateGenome = \
				probabilityThresholdToMutateGenome
	
	def ComputeNewPopulation(self, oldPopulation, fitnessList):
		parentPool = self._doSelection(oldPopulation, fitnessList)
		sizeOfPopulation = len(oldPopulation)
		sizeOfChromosome = len(oldPopulation[0])
		newPopulation = \
				self._doCrossover(parentPool, sizeOfPopulation, sizeOfChromosome)
		mutatedNewPopulation = self._doMutation(newPopulation)
		return torch.stack(mutatedNewPopulation)

	def _doSelection(self, oldPopulation, fitnessList):
		parentPool = []
		sizeOfPopulation = len(oldPopulation)
		sizeOfParentPool = self._computeSizeOfParentPool(sizeOfPopulation)
		contendersCount = self._computeContendersCount(sizeOfPopulation)
		for i in range(sizeOfParentPool):
			chromosomesToCompete = \
					self._getChromosomesToCompete(
							oldPopulation,
							fitnessList,
							contendersCount)
			bestChromosome = self._getBestChromosome(chromosomesToCompete)
			parentPool.append(bestChromosome)
		return parentPool

	def _doCrossover(self, parentPool, sizeOfNewPopulation, sizeOfChromosome):
		parentPairs = self._createParentPairs(parentPool)
		childrenPrefabsPool = []
		for i in range(len(parentPairs)):
			crossoverPoint = self._pickCrossoverPoint(sizeOfChromosome)
			firstChild, secondChild = \
					self._createChildrenPrefabs(parentPairs[i], crossoverPoint)
			childrenPrefabsPool.append(firstChild)
			childrenPrefabsPool.append(secondChild)
		newPopulation = \
				self._createNewPopulationFromPrefabs(
						childrenPrefabsPool,
						sizeOfNewPopulation)
		return newPopulation

	def _doMutation(self, populationToMutate):
		for i in range(len(populationToMutate)):
			tempRandom = random.random()
			if tempRandom >= self._probabilityThresholdToMutateChromosome:
				self._mutateChromosome(populationToMutate[i])
		return populationToMutate
	
	def _computeSizeOfParentPool(self, sizeOfPopulation):
		sizeOfParentPool = \
				int(sizeOfPopulation * self._selectionPercentRate / 100)
		if sizeOfParentPool < 1:
			sizeOfParentPool = 1
		return sizeOfParentPool
		
	def _computeContendersCount(self, sizeOfPopulation):
		from math import sqrt
		maxContendersCount = int(sqrt(sizeOfPopulation))
		contendersCount = random.randrange(maxContendersCount)
		if contendersCount < 1:
			contendersCount = 1
		return contendersCount

	def _getChromosomesToCompete(
			self,
			oldPopulation,
			fitnessScoreList,
			contendersCount):
		fitnessToChromosomesDict = dict(zip(fitnessScoreList, oldPopulation))
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
	
	def _createParentPairs(self, parentPool):
		sampledParentPool = random.sample(parentPool, len(parentPool))
		sizeOfPair = 2
		parentPairs = [
				sampledParentPool[i : i + sizeOfPair]
				for i in range(0, len(sampledParentPool), sizeOfPair)
		]
		return parentPairs
	
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

	def _createNewPopulationFromPrefabs(
			self,
			childrenPrefabs,
			sizeOfNewPopulation):
		from math import ceil
		cloneCounter = ceil(sizeOfNewPopulation / len(childrenPrefabs))
		newPopulation = childrenPrefabs * cloneCounter
		newPopulation = newPopulation[0 : sizeOfNewPopulation]
		return newPopulation
	
	def _mutateChromosome(self, chromosome):
		for i in range(len(chromosome)):
			tempRandom = random.random()
			if tempRandom >= self._probabilityThresholdToMutateGenome:
				chromosome[i] = tempRandom
