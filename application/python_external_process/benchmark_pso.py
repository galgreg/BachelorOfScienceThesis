from src.benchmark_functions import *
import torch
import random
import statistics

class Particle:
    def __init__(self, particleSize):
        self.position = torch.FloatTensor(particleSize).uniform_(-2.0, 2.0)
        self.pbest_position = self.position
        self.pbest_value = float("inf")
        self.velocity = torch.FloatTensor(particleSize).uniform_(-2.0, 2.0)
    
    def move(self):
        self.position = torch.clamp(self.position + self.velocity, -2.0, 2.0)

class SearchSpace:
    def __init__(
            self,
            swarmSize,
            particleSize,
            fitnessFunction,
            target,
            targetError = 0.0,
            W = 0.5,
            c1 = 0.8,
            c2 = 0.9):
        self.particles = [Particle(particleSize) for _ in range(swarmSize)]
        self.swarmSize = swarmSize
        self.fitnessFunction = fitnessFunction
        self.target = target
        self.targetError = targetError
        self.gbest_value = float("inf")
        self.gbest_position = torch.FloatTensor(particleSize).uniform_(-2.0, 2.0)
        self.W = W
        self.c1 = c1
        self.c2 = c2
        self.episodeCounter = 0
    
    def evaluate_pbest_and_gbest(self):
        fitnessList = []
        for particle in self.particles:
            fitnessCandidate = float(self.fitnessFunction(particle.position))
            fitnessList.append(fitnessCandidate)
            
            if particle.pbest_value > fitnessCandidate:
                particle.pbest_value = fitnessCandidate
                particle.pbest_position = particle.position
            
            if self.gbest_value > fitnessCandidate:
                self.gbest_value = fitnessCandidate
                self.gbest_position = particle.position
        
        self.printEpisodeStatistics(fitnessList)
        self.episodeCounter += 1
    
    def moveParticles(self):
        for particle in self.particles:
            firstAddend = self.W * particle.velocity
            secondAddend = self.c1 * random.random() * \
                    (particle.pbest_position - particle.position)
            thirdAddend = self.c2 * random.random() * \
                    (self.gbest_position - particle.position)
            
            newVelocity = firstAddend + secondAddend + thirdAddend
            particle.velocity = torch.clamp(newVelocity, -2.0, 2.0)
            particle.move()
    
    def printEpisodeStatistics(self, fitnessList):
        bestEpisodeFitness = min(fitnessList)
        meanFitness = statistics.mean(fitnessList)
        stdDevFitness = statistics.stdev(fitnessList)
        print("Episode {0}: globalBest = {1}, episodeBest = {2}, mean = {3}, " \
        "stdDev = {4}".format(
                self.episodeCounter,
                self.gbest_value,
                bestEpisodeFitness,
                meanFitness,
                stdDevFitness))
                

def main():
    FITNESS_FUNCTION = griewank
    RANDOM_SEED = 0
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    NUM_OF_EPISODES = 5000
    SWARM_SIZE = 50
    PARTICLE_SIZE = 32
    TARGET = 0.0
    TARGET_ERROR = 0.0
    W = 0.729
    c1 = 2.05
    c2 = 2.05
    
    searchSpace = SearchSpace(
            SWARM_SIZE,
            PARTICLE_SIZE,
            FITNESS_FUNCTION,
            TARGET,
            TARGET_ERROR,
            W,
            c1,
            c2)
    
    for episodeCounter in range(NUM_OF_EPISODES):
        searchSpace.evaluate_pbest_and_gbest()
        
        if abs(searchSpace.gbest_value - searchSpace.target) <= searchSpace.targetError:
            break
        
        searchSpace.moveParticles()
    

if __name__ == "__main__":
    main()
