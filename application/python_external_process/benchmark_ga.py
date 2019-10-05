from src.learning_algorithms.GeneticAlgorithm import *
from datetime import datetime
import torch
import math
import statistics

def ackley(chromosome):
    firstSum = 0.0
    secondSum = 0.0
    for genome in chromosome:
        firstSum += genome**2
        secondSum += math.cos(2.0 * math.pi * genome)
    
    n = len(chromosome)
    firstExp = math.exp(-0.2 * math.sqrt(firstSum / n))
    secondExp = math.exp(secondSum / n)
    
    functionResult = -20.0 * firstExp - secondExp + 20 + math.exp(1)
    if functionResult <= 5e-16:
        functionResult = float("inf")
    else:
        functionResult = 1 / functionResult
        
    return functionResult

def computeNewPopulation(algorithm, population, fitnessList):
    parentPool = algorithm._doSelection(population, fitnessList)
    sizeOfPopulation = len(population)
    sizeOfChromosome = len(population[0])
    newPopulation = \
            algorithm._doCrossover(parentPool, sizeOfPopulation, sizeOfChromosome)
    mutatedPopulation = algorithm._doMutation(newPopulation)
    return mutatedPopulation


def main():
    RANDOM_SEED = 0
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    NUM_OF_EPISODES = 500
    SIZE_OF_POPULATION = 100
    SIZE_OF_CHROMOSOME = 32 # docelowo 32
    population = torch.randn(SIZE_OF_POPULATION, SIZE_OF_CHROMOSOME)
    
    SELECTION_PERCENT_RATE = 10
    PROBABILITY_OF_CHROMOSOME_MUTATION = 1.0
    PROBABILITY_OF_GENOME_MUTATION = 0.02
    SCALE_OF_MUTATE_DEVIATION = 0.3
    algorithm = GeneticAlgorithm(
            SELECTION_PERCENT_RATE,
            PROBABILITY_OF_CHROMOSOME_MUTATION,
            PROBABILITY_OF_GENOME_MUTATION,
            SCALE_OF_MUTATE_DEVIATION)
    
    print("benchmark_ga.py parameters:\n" "NUM_OF_EPISODES = {0}, " \
            "SIZE_OF_POPULATION = {1}, SIZE_OF_CHROMOSOME = {2}, \n" \
            "SELECTION_PERCENT_RATE = {3}, " \
            "PROBABILITY_OF_CHROMOSOME_MUTATION = {4}, " \
            "PROBABILITY_OF_GENOME_MUTATION = {5}\n".format(
                    NUM_OF_EPISODES,
                    SIZE_OF_POPULATION,
                    SIZE_OF_CHROMOSOME,
                    SELECTION_PERCENT_RATE,
                    PROBABILITY_OF_CHROMOSOME_MUTATION,
                    PROBABILITY_OF_GENOME_MUTATION))

    
    for episodeCounter in range(NUM_OF_EPISODES):
        fitnessList = []
        for chromosome in population:
            tempFitness = ackley(chromosome)
            fitnessList.append(tempFitness)
        
        indexOfBestFitness = fitnessList.index(max(fitnessList))
        bestChromosome = population[indexOfBestFitness]
        bestFitness = fitnessList[indexOfBestFitness]
        
        inversedFitnessList = [1 / fitnessVal for fitnessVal in fitnessList]
        meanFitness = statistics.mean(inversedFitnessList)
        stdevFitness = statistics.stdev(inversedFitnessList)
        
        print("Episode: {0}, f(best_x) = {1}, mean = {2}, std_dev = {3}".format(
                episodeCounter,
                1 / bestFitness,
                meanFitness,
                stdevFitness))
        
        population = computeNewPopulation(algorithm, population, fitnessList)

if __name__ == "__main__":
    main()
