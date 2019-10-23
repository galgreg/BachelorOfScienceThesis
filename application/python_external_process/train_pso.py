from mlagents.envs import UnityEnvironment
from copy import deepcopy
from docopt import docopt
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
    train_pso.py <config-file-path> (--track-1 | --track-2 | --track-3) [options]
    train_pso.py -h | --help

Options:
    --track-1                               Run training on RaceTrack_1
    --track-2                               Run training on RaceTrack_2
    --track-3                               Run training on RaceTrack_3
    -v --verbose                            Run in verbose mode
    --save-population                       Save population after training
    --population=<pretrained-population>    Specify path to pretrained population
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    
    # --- 2 - Create logging object --- #
    trainingLog = TrainingLog(isVerbose = options["--verbose"])
    trainingLog.Append("Training log has been created!")
    trainingLog.Append("This is train_pso.py -> Particle Swarm Optimization " \
            "training!")
    
    if options["--track-1"]:
        trainingLog.Append("Training on RaceTrack_1.")
    elif options["--track-2"]:
        trainingLog.Append("Training on RaceTrack_2.")
    elif options["--track-3"]:
        trainingLog.Append("Training on RaceTrack_3.")
    
    # --- 3 - Load config data from file --- #
    pathToConfigFile = options["<config-file-path>"]
    CONFIG_DATA = loadConfigData(pathToConfigFile)
    trainingLog.Append("Config data has been loaded from file: {0}".format(
            pathToConfigFile))
    del pathToConfigFile
    
    # --- 4 - Set random seed --- #
    TRAINING_PARAMS = CONFIG_DATA["TrainingParameters"]
    RANDOM_SEED = TRAINING_PARAMS["randomSeed"]
    if isinstance(RANDOM_SEED, int):
        random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        trainingLog.Append("Random seed set to value: {0}".format(RANDOM_SEED))
    
    # --- 5 - Establish connection with Unity environment --- #
    env = UnityEnvironment()
    trainingLog.Append("Established connection to the Unity environment!")
   
    # --- 6 - Get info from Unity environment --- #
    brainName = env.brain_names[0]
    trainingLog.Append("Brain name: {0}".format(brainName))
    brain = env.brains[brainName]
    brainInfo = env.reset(train_mode=True)[brainName]
    observationSize = brainInfo.vector_observations.shape[1]
    actionSize = brain.vector_action_space_size[0]
    trainingLog.Append(
            "Loaded from Unity environment: observationSize = {0}, " \
            "actionSize = {1}".format(observationSize, actionSize))
    
    # --- 7 - Compute agent dimensions -- #
    HIDDEN_DIMENSIONS = TRAINING_PARAMS["networkHiddenDimensions"]
    agentDimensions = [observationSize - 1] + HIDDEN_DIMENSIONS + [actionSize]
    trainingLog.Append("Computed agentDimensions: {0}".format(agentDimensions))
    
    # --- 8 - Create population ---- #
    locationForPretrainedPopulation = options["--population"]
    PSO_PARAMS = CONFIG_DATA["LearningAlgorithms"]["pso"]
    NUM_OF_AGENTS = PSO_PARAMS["numberOfAgents"]
    resultsRepository = TrainingResultsRepository(trainingLog)
    
    if locationForPretrainedPopulation is None:
        population = [ AgentNeuralNetwork(agentDimensions) \
                for _ in range(NUM_OF_AGENTS) ]
        trainingLog.Append("Created new population, with parameters: " \
                "NUM_OF_AGENTS = {0}, agentDimensions = {1}, ".format(
                        NUM_OF_AGENTS,
                        agentDimensions))
    else:
        population = \
                resultsRepository.LoadPopulation(locationForPretrainedPopulation)
    
    # --- 9 - Training sequence --- #
    MAX_EPISODES_NUMBER = TRAINING_PARAMS["maxNumberOfEpisodes"]
    currentBestFitness = float("-inf")
    currentMeanFitness = float("-inf")
    currentStdDevFitness = float("-inf")
    
    try:
        if options["--track-1"]:
            MINIMAL_ACCEPTABLE_FITNESS = \
                    TRAINING_PARAMS["minimalAcceptableFitness"]["RaceTrack_1"]
        elif options["--track-2"]:
            MINIMAL_ACCEPTABLE_FITNESS = \
                    TRAINING_PARAMS["minimalAcceptableFitness"]["RaceTrack_2"]
        elif options["--track-3"]:
            MINIMAL_ACCEPTABLE_FITNESS = \
                    TRAINING_PARAMS["minimalAcceptableFitness"]["RaceTrack_3"]
        fitnessFunction = AgentFitnessEvaluator(env, brainName)
        
        trainingLog.Append("Start training with parameters: " \
                "MAX_EPISODES_NUMBER = {0}, " \
                "MINIMAL_ACCEPTABLE_FITNESS = {1}, " \
                "fitnessFunction = {2}".format(
                    MAX_EPISODES_NUMBER,
                    MINIMAL_ACCEPTABLE_FITNESS,
                    type(fitnessFunction)))
        
        particlePositions = retrieveParametersFromAgentList(population)
        particleVelocities = \
                torch.FloatTensor(particlePositions.shape).uniform_(-2.0, 2.0)
        
        pbestPositions = particlePositions.clone().detach()
        pbestFitnessValues = torch.tensor([float("-inf")] * NUM_OF_AGENTS)
        
        gbestPosition = torch.zeros(particlePositions.shape[1])
        gbestFitnessValue = float("-inf")
        
        W = PSO_PARAMS["W"]
        c1 = PSO_PARAMS["c1"]
        c2 = PSO_PARAMS["c2"]
        
        bestAgent = None
        for episodeCounter in range(MAX_EPISODES_NUMBER):
            fitnessList = []            
            for i in range(NUM_OF_AGENTS):
                fitnessCandidate = fitnessFunction(population[i])
                fitnessList.append(fitnessCandidate)
                
                if pbestFitnessValues[i] < fitnessCandidate:
                    pbestFitnessValues[i] = fitnessCandidate
                    pbestPositions[i] = particlePositions[i]
                
                if gbestFitnessValue < fitnessCandidate:
                    gbestFitnessValue = fitnessCandidate
                    gbestPosition = particlePositions[i]
                    bestAgent = deepcopy(population[i])
            
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
                setNewParametersOnAgent(population[i], particlePositions[i])
        
    except KeyboardInterrupt:
        trainingLog.Append("\nTraining interrupted because of KeyboardInterrupt!")
    
    trainingLog.Append("End of training!")
    
    # --- 10 - Close environment --- #
    env.close()
    trainingLog.Append("Closed Unity environment.")
    
    # --- 11 - Save training results --- #
    shouldSavePopulation = options["--save-population"]
    resultsRepository.Save(population, bestAgent, shouldSavePopulation)
    
if __name__ == "__main__":
    main()
