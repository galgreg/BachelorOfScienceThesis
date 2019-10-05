import random
import torch
from math import ceil, floor

class GeneticAlgorithm:
    def __init__(
            self,
            selectionPercentRate,
            probabilityThresholdToMutateChromosome,
            probabilityThresholdToMutateGenome,
            scaleOfMutateDeviation):
        self._selectionPercentRate = selectionPercentRate
        self._probabilityThresholdToMutateChromosome = \
                probabilityThresholdToMutateChromosome
        self._probabilityThresholdToMutateGenome = \
                probabilityThresholdToMutateGenome
        self._scaleOfMutateDeviation = scaleOfMutateDeviation
    
    def ComputeNewPopulation(self, agentList, fitnessList):
        agentsParameters = \
                self._retrieveParametersFromAgents(agentList)
        parentPool = self._doSelection(agentsParameters, fitnessList)
        sizeOfPopulation = len(agentsParameters)
        sizeOfChromosome = len(agentsParameters[0])
        newPopulationParameters = \
                self._doCrossover(parentPool, sizeOfPopulation, sizeOfChromosome)
        mutatedParameters = self._doMutation(newPopulationParameters)
        self._setNewParametersOnAgentList(agentList, mutatedParameters)
        return agentList
        
    def _retrieveParametersFromAgents(self, agentList):
        populationParameters = []
        for agent in agentList:
            agentParameters = []
            for layer in agent._layers:
                layerWeights = layer.weight
                numberOfWeights = layerWeights.numel()
                weightParameters = \
                        torch.reshape(layerWeights, (numberOfWeights,))
                agentParameters = \
                        agentParameters + weightParameters.tolist()
                agentParameters = \
                        agentParameters + layer.bias.tolist()
            
            populationParameters.append(agentParameters)
        
        return torch.tensor(populationParameters)

#### Selection method ########################
    def _doSelection(self, oldPopulation, fitnessList):
        return self._tournamentSelection(oldPopulation, fitnessList)
######## Selection subroutines #################################################
    def _tournamentSelection(self, population, fitnessListOriginal):
        fitnessList = fitnessListOriginal.copy()
        parentPool = []
        sizeOfPopulation = len(population)
        sizeOfParentPool = max(
                int(sizeOfPopulation * self._selectionPercentRate / 100),
                1)
        contendersCount = self._computeContendersCount(sizeOfPopulation)
        # contendersCount = 3
        for i in range(sizeOfParentPool - 1):
            chromosomesToCompete = \
                    self._getChromosomesToCompete(
                            population,
                            fitnessList,
                            contendersCount)
            bestChromosome = self._getBestChromosome(chromosomesToCompete)
            parentPool.append(bestChromosome)
        
        bestChromosomeInPopulation = self._getBestChromosome(
                dict(zip(fitnessList, population)))
        parentPool.append(bestChromosomeInPopulation)
        return parentPool

    def _selectPartWithBest(self, oldPopulation, fitnessListOriginal):
        fitnessList = fitnessListOriginal.copy()
        parentPool = []
        sizeOfPopulation = len(oldPopulation)
        sizeOfParentPool = max(
                int(sizeOfPopulation * self._selectionPercentRate / 100),
                1)
        for i in range(sizeOfParentPool):
            indexOfBestChromosome = fitnessList.index(max(fitnessList))
            bestChromosome = oldPopulation[indexOfBestChromosome]
            parentPool.append(bestChromosome)
            fitnessList[indexOfBestChromosome] = float("-inf")
        
        return parentPool

    def _rouletteWheelSelection(self, population, fitnessList):
        parentPool = []
        sizeOfParentPool = max(
                int(len(population) * self._selectionPercentRate / 100),
                1)
        fitnessSum = sum(fitnessList)
        for i in range(sizeOfParentPool):
            pick = random.uniform(0, fitnessSum)
            current = 0
            for chromosome, fitnessValue in zip(population, fitnessList):
                current += fitnessValue
                if current > pick:
                    parentPool.append(chromosome)
                    break
        return parentPool

#### Crossover method ##########################################################
    def _doCrossover(self, parentPool, sizeOfNewPopulation, sizeOfChromosome):
        childrenPool = self._breedPairsUntilNewPopulationIsReached_OnePointCrossover(
                parentPool,
                sizeOfNewPopulation,
                sizeOfChromosome)
        newPopulation = parentPool + childrenPool[len(parentPool) : ]
        return newPopulation

######## Crossover subroutines #################################################
    def _breedPairsUntilNewPopulationIsReached_OnePointCrossover(
            self,
            parentPool,
            sizeOfNewPopulation,
            sizeOfChromosome):
        newPopulation = []
        sizeOfParentPool = len(parentPool)
        
        for i in range(int(sizeOfNewPopulation / 2)):
        # for i in range(floor(sizeOfNewPopulation / 5 * 2)):
            firstParentIndex = random.randrange(sizeOfParentPool)
            secondParentIndex = random.randrange(sizeOfParentPool)
            parentPair = [ \
                    parentPool[firstParentIndex], \
                    parentPool[secondParentIndex] \
            ]
            crossoverPoint = self._pickCrossoverPoint(sizeOfChromosome)
            firstChild, secondChild = \
                    self._breedParentPair(parentPair, crossoverPoint)
            newPopulation.append(firstChild)
            newPopulation.append(secondChild)
        
        # for i in range(ceil(sizeOfNewPopulation / 5)):
            # randomChild = torch.randn(sizeOfChromosome)
            # newPopulation.append(randomChild)
        
        return newPopulation

    def _breedPairsUntilNewPopulationIsReached_UniformCrossover(
            self,
            parentPool,
            sizeOfNewPopulation,
            sizeOfChromosome):
        newPopulation = []
        sizeOfParentPool = len(parentPool)
        
        for i in range(int(sizeOfNewPopulation / 2)):
            firstParentIndex = random.randrange(sizeOfParentPool)
            secondParentIndex = random.randrange(sizeOfParentPool)
            parentPair = [ \
                    parentPool[firstParentIndex], \
                    parentPool[secondParentIndex] \
            ]
            firstChild, secondChild = \
                    self._shufflePairGenomesToProduceChildren(parentPair)
            newPopulation.append(firstChild)
            newPopulation.append(secondChild)
        
        return newPopulation

    def _uniformCrossover_OnlyOneChildPerCrossover_SomeChildrenRandom(
            self,
            parentPool,
            sizeOfNewPopulation,
            sizeOfChromosome):
        newPopulation = []
        sizeOfParentPool = len(parentPool)
        
        NUMBER_OF_RANDOM_CHILDREN = 5
        
        for i in range(sizeOfNewPopulation - len(parentPool) - NUMBER_OF_RANDOM_CHILDREN):
            firstParentIndex = random.randrange(sizeOfParentPool)
            secondParentIndex = random.randrange(sizeOfParentPool)
            parentPair = [ \
                    parentPool[firstParentIndex], \
                    parentPool[secondParentIndex] \
            ]
            firstChild, secondChild = \
                    self._shufflePairGenomesToProduceChildren(parentPair)
            newPopulation.append(firstChild)
        
        for i in range(NUMBER_OF_RANDOM_CHILDREN):
            newPopulation.append(torch.randn(sizeOfChromosome))
        
        return newPopulation
#### Auxiliary function for _breedPairsUntilNewPopulationIsReached() ###########
    def _shufflePairGenomesToProduceChildren(self, parents):
        if len(parents) == 1:
            return parents * 2
        elif len(parents) == 2:
            firstParent = parents[0]
            secondParent = parents[1]
            firstChild = torch.zeros(len(firstParent))
            secondChild = torch.zeros(len(firstParent))
            for i in range(len(firstParent)):
                if random.random() < 0.5:
                    firstChild[i] = firstParent[i]
                    secondChild[i] = secondParent[i]
                else:
                    firstChild[i] = secondParent[i]
                    secondChild[i] = firstParent[i]
            
            return [firstChild, secondChild]
        else:
            raise ValueError("GeneticAlgorithm._breedParentPair error: "
                    "parents has invalid size! (should be 1 or 2)!")
################################################################################
    def _doMutation(self, populationToMutate):
        for i in range(len(populationToMutate)):
            tempRandom = random.random()
            if tempRandom <= self._probabilityThresholdToMutateChromosome:
                self._mutateChromosome(populationToMutate[i])
        return populationToMutate
    
    def _setNewParametersOnAgentList(self, agentList, newParameters):
        for agent, agentParameters in zip(agentList, newParameters):
            agentParameters = agentParameters.tolist()
            for layer in agent._layers:
                numberOfWeights = layer.weight.numel()
                numberOfBiases = layer.bias.numel()
                
                weightParameters = \
                        torch.tensor( agentParameters[ : numberOfWeights] )
                weightDimensions = tuple(layer.weight.size())
                reshapedWeightParameters = \
                        torch.reshape(weightParameters, weightDimensions)
                layer.weight.data = reshapedWeightParameters
                del agentParameters[ : numberOfWeights ]
                
                biasParameters = \
                        torch.tensor( agentParameters[ : numberOfBiases] )
                layer.bias.data = biasParameters
                del agentParameters[ : numberOfBiases ]

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

    def _breedParentPair(self, parents, crossOverPoint):
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
            raise ValueError("GeneticAlgorithm._breedParentPair error: "
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
            if tempRandom <= self._probabilityThresholdToMutateGenome:
                chromosome[i] += \
                        float(torch.randn(1)) * self._scaleOfMutateDeviation
