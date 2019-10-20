from mlagents.envs import UnityEnvironment
from copy import deepcopy
from docopt import docopt
from src.AgentsPopulation import *
from src.training.TrainingLog import *
from src.training.TrainingResultsRepository import *
from src.training.training_utilities import *
import statistics
import random

def main():
    # --- 1 - Specify script's usage options --- #
    APP_USAGE_DESCRIPTION = """
Train neural networks to drive a car on a racetrack. Racetrack must be valid Unity ML-Agents environment.
Algorithm used to train is Particle Swarm Optimization.

Usage:
    train_pso.py [options]
    train_pso.py -h | --help

Options:
    -v --verbose                            Run in verbose mode
    --save-population                       Save population after training
    --population=<pretrained-population>    Specify path to pretrained population
    --random-seed=<random_seed>             Specify random seed
    --max-episodes-number=<n>               Specify max number of episodes for training [default: 1000]
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    
    # --- 2 - Create logging object --- #
    trainingLog = TrainingLog(isVerbose = options["--verbose"])
    trainingLog.Append("Training log has been created!")
    trainingLog.Append("This is train_pso.py -> Particle Swarm Optimization " \
            "training!")
    
    # --- 3 - Set random seed --- #
    RANDOM_SEED = options["--random-seed"]
    if isinstance(RANDOM_SEED, str) and RANDOM_SEED.isdigit():
        RANDOM_SEED = int(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        trainingLog.Append("Random seed set to value: {0}".format(RANDOM_SEED))
    
    # --- 4 - Establish connection with Unity environment --- #
    env = UnityEnvironment()
    trainingLog.Append("Established connection to the Unity environment!")
   
    # --- 5 - Get info from Unity environment --- #
    brainName = env.brain_names[0]
    trainingLog.Append("Brain name: {0}".format(brainName))
    brain = env.brains[brainName]
    brainInfo = env.reset(train_mode=True)[brainName]
    observationSize = brainInfo.vector_observations.shape[1]
    actionSize = brain.vector_action_space_size[0]
    trainingLog.Append(
            "Loaded from Unity environment: observationSize = {0}, " \
            "actionSize = {1}".format(observationSize, actionSize))
    
    # --- 6 - Compute agent dimensions -- #
    HIDDEN_DIMENSIONS = [5 ]
    agentDimensions = [observationSize - 1] + HIDDEN_DIMENSIONS + [actionSize]
    trainingLog.Append("Computed agentDimensions: {0}".format(agentDimensions))
    
    # --- 7 - Create population ---- #
    locationForPretrainedPopulation = options["--population"]
    NUM_OF_AGENTS = 100
    population = None
    resultsRepository = TrainingResultsRepository(trainingLog)
    
    if locationForPretrainedPopulation is None:
        population = AgentsPopulation(NUM_OF_AGENTS, agentDimensions, None)
        trainingLog.Append("Created new population, with parameters: " \
                "NUM_OF_AGENTS = {0}, agentDimensions = {1}, ".format(
                        NUM_OF_AGENTS,
                        agentDimensions))
    else:
        population = \
                resultsRepository.LoadPopulation(locationForPretrainedPopulation)
        population._sizeOfOutput = agentDimensions[-1]
        population._learningAlgorithm = None
    
    # --- 8 - Training sequence --- #
    MAX_EPISODES_NUMBER = int(options["--max-episodes-number"])
    currentBestFitness = float("-inf")
    currentMeanFitness = float("-inf")
    currentStdDevFitness = float("-inf")
    
    try:
        MINIMAL_ACCEPTABLE_FITNESS = 104.0 # RaceTrack_1
        # MINIMAL_ACCEPTABLE_FITNESS = 110.0 # RaceTrack_2
        # MINIMAL_ACCEPTABLE_FITNESS = 130.0 # RaceTrack_3
        fitnessFunction = AgentFitnessEvaluator(env, brainName)
        
        trainingLog.Append("Start training with parameters: " \
                "MAX_EPISODES_NUMBER = {0}, " \
                "MINIMAL_ACCEPTABLE_FITNESS = {1}, " \
                "fitnessFunction = {2}".format(
                    MAX_EPISODES_NUMBER,
                    MINIMAL_ACCEPTABLE_FITNESS,
                    type(fitnessFunction)))
        
        ########### Start of Particle Swarm Optimization #######################
        particlePositions = retrieveParametersFromAgentList(population._agents)
        particleVelocities = \
                torch.FloatTensor(particlePositions.shape).uniform_(-2.0, 2.0)
        
        pbestPositions = particlePositions.clone().detach()
        pbestFitnessValues = torch.tensor([float("-inf")] * NUM_OF_AGENTS)
        
        gbestPosition = torch.zeros(particlePositions.shape[1])
        gbestFitnessValue = float("-inf")
        
        W = 0.729
        c1 = 2.05
        c2 = 2.05
        
        bestAgent = None
        for episodeCounter in range(MAX_EPISODES_NUMBER):
            fitnessList = []            
            for i in range(NUM_OF_AGENTS):
                fitnessCandidate = fitnessFunction(population._agents[i])
                fitnessList.append(fitnessCandidate)
                
                if pbestFitnessValues[i] < fitnessCandidate:
                    pbestFitnessValues[i] = fitnessCandidate
                    pbestPositions[i] = particlePositions[i]
                
                if gbestFitnessValue < fitnessCandidate:
                    gbestFitnessValue = fitnessCandidate
                    gbestPosition = particlePositions[i]
                    bestAgent = deepcopy(population._agents[i])
            
            bestEpisodeFitness = max(fitnessList)
            meanFitness = statistics.mean(fitnessList)
            stdDevFitness = statistics.stdev(fitnessList)
            trainingLog.Append(
                    "Episode {0}: globalBest = {1}, episodeBest = {2}, " \
                    "mean = {3}, stdDev = {4}".format(
                            episodeCounter,
                            gbestFitnessValue,
                            bestEpisodeFitness,
                            meanFitness,
                            stdDevFitness))
            
            if gbestFitnessValue >= MINIMAL_ACCEPTABLE_FITNESS:
                trainingLog.Append(
                        "Training interrupted after {0} episodes, reason: " \
                        "reached minimal acceptable value for globalBestFitness!" \
                        " (minimalAcceptableFitness = {1}, globalBestFitness = {2})" \
                        .format(
                                episodeCounter + 1,
                                MINIMAL_ACCEPTABLE_FITNESS,
                                gbestFitnessValue))
                break
            
            for i in range(NUM_OF_AGENTS):
                firstAddend = W * particleVelocities[i]
                secondAddend = c1 * random.random() * \
                        (pbestPositions[i] - particlePositions[i])
                thirdAddend = c2 * random.random() * \
                        (gbestPosition - particlePositions[i])
                
                newVelocity = firstAddend + secondAddend + thirdAddend
                particleVelocities[i] = torch.clamp(newVelocity, -2.0, 2.0)
                particlePositions[i] = torch.clamp(
                        particlePositions[i] + particleVelocities[i],
                        -2.0, 2.0)
                setNewParametersOnAgent(population._agents[i], particlePositions[i])
        
    except KeyboardInterrupt:
        trainingLog.Append("Training interrupted because of KeyboardInterrupt!")
    
    trainingLog.Append("End of training!")
    
    # --- 9 - Close environment --- #
    env.close()
    trainingLog.Append("Closed Unity environment.")
    
    # --- 10 - Save training results --- #
    shouldSavePopulation = options["--save-population"]
    resultsRepository.Save(population, bestAgent, shouldSavePopulation)
    
if __name__ == "__main__":
    main()
