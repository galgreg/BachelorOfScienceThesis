from mlagents.envs import UnityEnvironment
from copy import deepcopy
from docopt import docopt
from src.training.TrainingLog import *
from src.training.TrainingResultsRepository import *
from src.training.training_utilities import *
import statistics
import sys
import random

def train_de():
    # --- 1 - Specify script's usage options --- #
    APP_USAGE_DESCRIPTION = """
Train neural networks to drive a car on a racetrack. Racetrack must be valid Unity ML-Agents environment.
Algorithm used to train is Differential Evolution.

Usage:
    train_de.py <config-file-path> (--track-1 | --track-2 | --track-3) [options]
    train_de.py -h | --help

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
    trainingLog.Append("This is train_de.py -> Differential Evolution training!")
    
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
    DIFF_EVO_PARAMS = CONFIG_DATA["LearningAlgorithms"]["diff_evo"]
    NUM_OF_AGENTS = DIFF_EVO_PARAMS["numberOfAgents"]
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
    
    bestAgent = None
    try:
        MUTATION_FACTOR = DIFF_EVO_PARAMS["mutationFactor"]
        CROSS_PROBABILITY = DIFF_EVO_PARAMS["crossProbability"]
        NUM_OF_PARAMS = computeNumOfParameters(agentDimensions)
        
        if options["--track-1"]:
            MINIMAL_ACCEPTABLE_FITNESS = \
                    TRAINING_PARAMS["minimalAcceptableFitness"]["RaceTrack_1"]
        elif options["--track-2"]:
            MINIMAL_ACCEPTABLE_FITNESS = \
                    TRAINING_PARAMS["minimalAcceptableFitness"]["RaceTrack_2"]
        elif options["--track-3"]:
            MINIMAL_ACCEPTABLE_FITNESS = \
                    TRAINING_PARAMS["minimalAcceptableFitness"]["RaceTrack_3"]
        
        fitnessList = []
        fitnessEvaluation = AgentFitnessEvaluator(env, brainName)
        
        trainingLog.Append(
            "Start training with parameters: MAX_EPISODES_NUMBER = {0}, " \
            "MUTATION_FACTOR = {1}, CROSS_PROBABILITY = {2}, NUM_OF_PARAMS = " \
            "{3}, fitnessFunction = {4}".format(
                    MAX_EPISODES_NUMBER,
                    MUTATION_FACTOR,
                    CROSS_PROBABILITY,
                    NUM_OF_PARAMS,
                    type(fitnessEvaluation)))
        
        for agentIndex in range(NUM_OF_AGENTS):
            agentFitness = fitnessEvaluation(population[agentIndex])
            fitnessList.append(agentFitness)

        bestFitness = max(fitnessList)
        indexOfBestFitness = fitnessList.index(bestFitness)
        bestAgent = deepcopy(population[indexOfBestFitness])
        meanFitness = statistics.mean(fitnessList)
        stdDevFitness = statistics.stdev(fitnessList)
        
        pop_denorm = retrieveParametersFromAgentList(population)
        pop_norm = pop_denorm / 4 + 0.5
        for i in range(MAX_EPISODES_NUMBER):
            for j in range(NUM_OF_AGENTS):
                indices = [index for index in range(NUM_OF_AGENTS) if index != j]
                a_idx, b_idx, c_idx = random.sample(indices, 3)
                a, b, c = pop_norm[a_idx], pop_norm[b_idx], pop_norm[c_idx]
                mutant = torch.clamp(a + MUTATION_FACTOR * (b - c), 0.0, 1.0)
                cross_points = [random.uniform(0, 1) for _ in range(NUM_OF_PARAMS)]

                trial_norm = torch.zeros(NUM_OF_PARAMS)
                for k in range(NUM_OF_PARAMS):
                    if cross_points[k] < CROSS_PROBABILITY:
                        trial_norm[k] = mutant[k]
                    else:
                        trial_norm[k] = pop_norm[j][k]
                trial_denorm = trial_norm * 4 - 2

                setNewParametersOnAgent(population[j], trial_denorm)
                agentIndex = j
                fitness_trial = fitnessEvaluation(population[agentIndex])
                
                if fitness_trial > fitnessList[j]:
                    fitnessList[j] = fitness_trial
                    pop_denorm[j] = trial_denorm
                    pop_norm[j] = trial_norm
                    if fitness_trial > bestFitness:
                        bestFitness = fitness_trial
                        bestAgent = deepcopy(population[j])
                else:
                    setNewParametersOnAgent(population[j], pop_denorm[j])
                
            meanFitness = statistics.mean(fitnessList)
            stdDevFitness = statistics.stdev(fitnessList)
            trainingLog.Append(
                    "Episode {0}: best = {1}, mean = {2}, stdDev = {3}".format(
                        i, bestFitness, meanFitness, stdDevFitness))
            
            if bestFitness >= MINIMAL_ACCEPTABLE_FITNESS:
                trainingLog.Append(
                        "Training interrupted after {0} episodes, reason: " \
                        "reached minimal acceptable value for bestFitness!" \
                        " (minimalAcceptableFitness = {1}, bestFitness = {2})" \
                        .format(i + 1, MINIMAL_ACCEPTABLE_FITNESS, bestFitness))
                break
        
    except:
        trainingLog.Append("Training interrupted because of exception: {0}" \
                .format(sys.exc_info()[0]))
    
    trainingLog.Append("End of training!")
    
    # --- 10 - Close environment --- #
    env.close()
    trainingLog.Append("Closed Unity environment.")
    
    # --- 11 - Save training results --- #
    shouldSavePopulation = options["--save-population"]
    resultsRepository.Save(population, bestAgent, shouldSavePopulation)
    
if __name__ == "__main__":
    train_de()
