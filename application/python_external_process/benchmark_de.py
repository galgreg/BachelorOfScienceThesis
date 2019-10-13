import math
import numpy as np
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
        functionResult = 0
    
    return functionResult

def griewank(chromosome):
    part1 = 0
    for i in range(len(chromosome)):
        part1 += chromosome[i]**2
        part2 = 1
    for i in range(len(chromosome)):
        part2 *= math.cos(float(chromosome[i]) / math.sqrt(i+1))
    return 1 + (float(part1)/4000.0) - float(part2)

def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=500):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    
    bestFitness = float("-inf")
    meanFitness = float("-inf")
    stdDevFitness = float("-inf")
    
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        
        bestFitness = fitness[best_idx]
        meanFitness = np.mean(fitness)
        stdDevFitness = np.std(fitness)
        print("Episode {0}: best = {1}, mean = {2}, stdDev = {3}".format(
                i, bestFitness, meanFitness, stdDevFitness))
        if bestFitness == 0.0 or stdDevFitness == 0.0:
            return

def main():
    FITNESS_FUNCTION = griewank
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)
    
    NUM_OF_EPISODES = 1000000
    SIZE_OF_POPULATION = 20
    SIZE_OF_CHROMOSOME = 100
    PARAMETER_BOUNDS = [(-2.0, 2.0)] * SIZE_OF_CHROMOSOME

    de(FITNESS_FUNCTION, PARAMETER_BOUNDS, mut = 0.8, crossp = 0.7,
        popsize = SIZE_OF_POPULATION, its = NUM_OF_EPISODES)

if __name__ == "__main__":
    main()
