from mlagents.envs import UnityEnvironment
from copy import deepcopy
from docopt import docopt
from src.Logger import *
from src.experiment.ExperimentDataCollector import *
from src.training.TrainingResultsRepository import *
from src.training.training_utilities import *
import statistics
import random
import time

def getProgramOptions():
    APP_USAGE_DESCRIPTION = """
Train neural networks to drive a car on a racetrack. Racetrack must be valid Unity ML-Agents environment.
Algorithm used to train is Differential Evolution.
NOTE: As a config file should be used 'config.json' file or other with appropriate fields.

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
    --env-path=<unity-build>                Specify path to Unity environment build
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    return options

def train_de(options, trainingLog, dataCollector = None):
    isTrainInExperimentMode = isinstance(dataCollector, ExperimentDataCollector)
    trainingLog.Append("This is train_de.py -> Differential Evolution training!")
    
    if options["--track-1"]:
        trackNumber = 1
    elif options["--track-2"]:
        trackNumber = 2
    elif options["--track-3"]:
        trackNumber = 3
    
    trainingLog.Append("Training on RaceTrack_{0}.".format(trackNumber))
    
    # --- Load config data from file --- #
    pathToConfigFile = options["<config-file-path>"]
    CONFIG_DATA = loadConfigData(pathToConfigFile)
    trainingLog.Append("Config data has been loaded from file: {0}".format(
            pathToConfigFile))
    del pathToConfigFile
    
    # --- Set random seed --- #
    TRAINING_PARAMS = CONFIG_DATA["TrainingParameters"]
    RANDOM_SEED = TRAINING_PARAMS["randomSeed"]
    if isinstance(RANDOM_SEED, int):
        random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        trainingLog.Append("Random seed set to value: {0}".format(RANDOM_SEED))
    
    # --- Establish connection with Unity environment --- #
    pathToEnv = options["--env-path"]
    env = UnityEnvironment(file_name = pathToEnv)
    
    if pathToEnv is None:
        trainingLog.Append("Established connection with Unity Editor!")
    else:
        trainingLog.Append("Established connection with Unity build '{0}'!" \
                .format(pathToEnv))
    del pathToEnv
   
    # --- Get info from Unity environment --- #
    brainName = env.brain_names[0]
    trainingLog.Append("Brain name: {0}".format(brainName))
    brain = env.brains[brainName]
    brainInfo = env.reset(train_mode=True)[brainName]
    observationSize = brainInfo.vector_observations.shape[1]
    actionSize = brain.vector_action_space_size[0]
    trainingLog.Append(
            "Loaded from Unity environment: observationSize = {0}, " \
            "actionSize = {1}".format(observationSize, actionSize))
    
    # --- Compute agent dimensions -- #
    HIDDEN_DIMENSIONS = TRAINING_PARAMS["networkHiddenDimensions"]
    agentDimensions = [observationSize - 1] + HIDDEN_DIMENSIONS + [actionSize]
    trainingLog.Append("Computed agentDimensions: {0}".format(agentDimensions))
    
    # --- Create population ---- #
    locationForPretrainedPopulation = options["--population"]
    DIFF_EVO_PARAMS = CONFIG_DATA["LearningAlgorithms"]["diff_evo"]
    NUM_OF_AGENTS = DIFF_EVO_PARAMS["numberOfAgents"]
    resultsRepository = TrainingResultsRepository(trainingLog)

    try:
        MUTATION_FACTOR = DIFF_EVO_PARAMS["mutationFactor"]
        CROSS_PROBABILITY = DIFF_EVO_PARAMS["crossProbability"]
        NUM_OF_PARAMS = computeNumOfParameters(agentDimensions)
        
        trackName = "RaceTrack_{0}".format(trackNumber)
        MINIMAL_ACCEPTABLE_FITNESS = \
                TRAINING_PARAMS["minimalAcceptableFitness"][trackName]
        del trackName
        
        fitnessEvaluation = AgentFitnessEvaluator(env, brainName)
        
        MAX_EPISODES_NUMBER = TRAINING_PARAMS["maxNumberOfEpisodes"]
        MAX_REPEATS_NUMBER = TRAINING_PARAMS["maxNumberOfRepeatsIfTrainingFails"]
        trainingLog.Append(
            "Start training with parameters: MAX_EPISODES_NUMBER = {0}, " \
            "MAX_REPEATS_NUMBER = {1}, MUTATION_FACTOR = {2}, " \
            "CROSS_PROBABILITY = {3}, NUM_OF_PARAMS = {4}, " \
            "MINIMAL_ACCEPTABLE_FITNESS = {5}, fitnessFunction = {6}".format(
                    MAX_EPISODES_NUMBER,
                    MAX_REPEATS_NUMBER,
                    MUTATION_FACTOR,
                    CROSS_PROBABILITY,
                    NUM_OF_PARAMS,
                    MINIMAL_ACCEPTABLE_FITNESS,
                    type(fitnessEvaluation)))
        
        shouldContinueTraining = True
        for repeatCounter in range(MAX_REPEATS_NUMBER):
            if not shouldContinueTraining:
                break
            
            if locationForPretrainedPopulation is None:
                population = [ AgentNeuralNetwork(agentDimensions) \
                        for _ in range(NUM_OF_AGENTS) ]
                trainingLog.Append("Created new population, with parameters: " \
                        "NUM_OF_AGENTS = {0}, agentDimensions = {1}, ".format(
                                NUM_OF_AGENTS,
                                agentDimensions))
            else:
                population = \
                        resultsRepository.LoadPopulation(
                                locationForPretrainedPopulation)
                if population is None:
                    env.close()
                    exit()
            
            fitnessList = []
            if isTrainInExperimentMode:
                bestFitnessSequence = []
                meanFitnessSequence = []
                stdevFitnessSequence = []
                searchCounter = NUM_OF_AGENTS
                timeOfBegin = time.time()
            
            for agentIndex in range(NUM_OF_AGENTS):
                agentFitness = fitnessEvaluation(population[agentIndex])
                fitnessList.append(agentFitness)

            bestFitness = max(fitnessList)
            indexOfBestFitness = fitnessList.index(bestFitness)
            bestAgent = deepcopy(population[indexOfBestFitness])
            meanFitness = statistics.mean(fitnessList)
            stdDevFitness = statistics.stdev(fitnessList)
            
            if isTrainInExperimentMode:
                bestFitnessSequence.append(bestFitness),
                meanFitnessSequence.append(meanFitness)
                stdevFitnessSequence.append(stdDevFitness)
            
            pop_denorm = retrieveParametersFromAgentList(population)
            pop_norm = pop_denorm / 4 + 0.5
            for episodeCounter in range(MAX_EPISODES_NUMBER):
                for j in range(NUM_OF_AGENTS):
                    if isTrainInExperimentMode:
                        searchCounter += 1
                    
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
                
                if isTrainInExperimentMode:
                    bestFitnessSequence.append(bestFitness),
                    meanFitnessSequence.append(meanFitness)
                    stdevFitnessSequence.append(stdDevFitness)
                
                trainingLog.Append(
                        "Episode {0}: best = {1}, mean = {2}, stdDev = {3}".format(
                            episodeCounter, bestFitness, meanFitness, stdDevFitness))
                
                if bestFitness >= MINIMAL_ACCEPTABLE_FITNESS:
                    trainingLog.Append(
                            "Training interrupted after {0} episodes, reason: " \
                            "reached minimal acceptable value for bestFitness!" \
                            " (minimalAcceptableFitness = {1}, bestFitness = {2})" \
                            .format(
                                    episodeCounter + 1,
                                    MINIMAL_ACCEPTABLE_FITNESS,
                                    bestFitness))
                    
                    if isTrainInExperimentMode:
                        timeOfEnd = time.time()
                        trainingTime = timeOfEnd - timeOfBegin
                        
                        dataCollector.AppendBestFitnessSequence(
                                trackNumber,
                                "DE",
                                bestFitnessSequence)
                        dataCollector.AppendMeanFitnessSequence(
                                trackNumber,
                                "DE",
                                meanFitnessSequence)
                        dataCollector.AppendStdevFitnessSequence(
                                trackNumber,
                                "DE",
                                stdevFitnessSequence)
                        dataCollector.AddTimeInSecondsFromTraining(
                                trackNumber,
                                "DE",
                                trainingTime)
                        trainingLog.Append("Training time in seconds: {0}" \
                                .format(trainingTime))
                        dataCollector.AddTimeInEpisodesFromTraining(
                                trackNumber,
                                "DE",
                                episodeCounter + 1)
                        trainingLog.Append("Training time in episodes: {0}" \
                                .format(episodeCounter + 1))
                        dataCollector.AddToSearchCounter(
                                trackNumber,
                                "DE",
                                searchCounter)
                        trainingLog.Append(
                                "searchCounter = {0}".format(searchCounter))
                    shouldContinueTraining = False
                    break
                
                if episodeCounter >= (MAX_EPISODES_NUMBER - 1):
                    message = "Cannot train population in current repeat " \
                            "(MAX_EPISODES_NUMBER = {0}, repeatCounter = {1})!" \
                            .format(MAX_EPISODES_NUMBER, repeatCounter)
                    if repeatCounter < MAX_REPEATS_NUMBER - 1:
                        message += " Try again to train population!"
                    else:
                        message += " Unfortunately, cannot try again. " \
                                "Reason: achieved maximum number of repeats!"
                    trainingLog.Append(message)
    
    except KeyboardInterrupt:
        trainingLog.Append("\nTraining interrupted because of KeyboardInterrupt!")
    
    trainingLog.Append("End of training!")
    
    # --- Close environment --- #
    env.close()
    trainingLog.Append("Closed Unity environment.")
    
    # --- Save training results --- #
    shouldSavePopulation = options["--save-population"]
    resultsRepository.Save(population, bestAgent, shouldSavePopulation)
    
    if isTrainInExperimentMode:
        dataCollector.PathToLastSavedModel = \
                resultsRepository._pathToLastSavedModel
    
if __name__ == "__main__":
    options = getProgramOptions()
    trainingLog = Logger(isVerbose = options["--verbose"])
    trainingLog.Append("Training log has been created!")
    train_de(options, trainingLog)
