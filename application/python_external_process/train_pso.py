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
Algorithm used to train is Particle Swarm Optimization.
NOTE: As a config file should be used 'config.json' file or other with appropriate fields.

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
    --env-path=<unity-build>                Specify path to Unity environment build
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    return options

def train_pso(options, trainingLog, dataCollector = None):
    isTrainInExperimentMode = isinstance(dataCollector, ExperimentDataCollector)
    trainingLog.Append("This is train_pso.py -> Particle Swarm Optimization " \
            "training!")
    
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
    env = UnityEnvironment(file_name = options["--env-path"])
    trainingLog.Append("Established connection to the Unity environment!")
   
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
        if population is None:
            env.close()
            exit()
    
    # --- Training sequence --- #
    MAX_EPISODES_NUMBER = TRAINING_PARAMS["maxNumberOfEpisodes"]
    currentBestFitness = float("-inf")
    currentMeanFitness = float("-inf")
    currentStdDevFitness = float("-inf")
    
    try:
        trackName = "RaceTrack_{0}".format(trackNumber)
        MINIMAL_ACCEPTABLE_FITNESS = \
                TRAINING_PARAMS["minimalAcceptableFitness"][trackName]
        del trackName

        W = PSO_PARAMS["W"]
        c1 = PSO_PARAMS["c1"]
        c2 = PSO_PARAMS["c2"]
        fitnessFunction = AgentFitnessEvaluator(env, brainName)
        
        trainingLog.Append("Start training with parameters: " \
                "MAX_EPISODES_NUMBER = {0}, MINIMAL_ACCEPTABLE_FITNESS = {1}," \
                " W = {2}, c1 = {3}, c2 = {4}, fitnessFunction = {5}".format(
                    MAX_EPISODES_NUMBER,
                    MINIMAL_ACCEPTABLE_FITNESS,
                    W,
                    c1,
                    c2,
                    type(fitnessFunction)))
        
        particlePositions = retrieveParametersFromAgentList(population)
        particleVelocities = \
                torch.FloatTensor(particlePositions.shape).uniform_(-2.0, 2.0)
        
        pbestPositions = particlePositions.clone().detach()
        pbestFitnessValues = [float("-inf")] * NUM_OF_AGENTS
        
        gbestPosition = torch.zeros(particlePositions.shape[1])
        gbestFitnessValue = float("-inf")
        
        bestAgent = None
        
        if isTrainInExperimentMode:
            gbestSequence = []
            pbestMeanSequence = []
            episodeMeanSequence = []
            pbestStdevSequence = []
            episodeStdevSequence = []
            timeOfBegin = time.time()
            searchCounter = 0
        
        for episodeCounter in range(MAX_EPISODES_NUMBER):
            fitnessList = []            
            for i in range(NUM_OF_AGENTS):
                if isTrainInExperimentMode:
                    searchCounter += 1
                
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
            pbestMeanFitness = statistics.mean(pbestFitnessValues)
            episodeMeanFitness = statistics.mean(fitnessList)
            pbestStdevFitness = statistics.stdev(pbestFitnessValues)
            episodeStdevFitness = statistics.stdev(fitnessList)
            trainingLog.Append(
                    "Episode {0}: globalBest = {1}, episodeBest = {2}, " \
                    "mean = {3}, stdDev = {4}".format(
                            episodeCounter,
                            gbestFitnessValue,
                            bestEpisodeFitness,
                            episodeMeanFitness,
                            episodeStdevFitness))
            
            if isTrainInExperimentMode:
                gbestSequence.append(gbestFitnessValue)
                pbestMeanSequence.append(pbestFitnessValues)
                episodeMeanSequence.append(episodeMeanFitness)
                pbestStdevSequence.append(pbestStdevFitness)
                episodeStdevSequence.append(episodeStdevFitness)
            
            if gbestFitnessValue >= MINIMAL_ACCEPTABLE_FITNESS:
                trainingLog.Append(
                        "Training interrupted after {0} episodes, reason: " \
                        "reached minimal acceptable value for globalBestFitness!" \
                        " (minimalAcceptableFitness = {1}, globalBestFitness = {2})" \
                        .format(
                                episodeCounter + 1,
                                MINIMAL_ACCEPTABLE_FITNESS,
                                gbestFitnessValue))
                
                if isTrainInExperimentMode:
                    timeOfEnd = time.time()
                    trainingTime = timeOfEnd - timeOfBegin
                    
                    dataCollector.AppendBestFitnessSequence(
                            trackNumber,
                            "PSO",
                            gbestSequence)
                    dataCollector.AppendMeanFitnessSequence(
                            trackNumber,
                            "PSO_Pbest",
                            pbestMeanSequence)
                    dataCollector.AppendMeanFitnessSequence(
                            trackNumber,
                            "PSO_Episode",
                            episodeMeanSequence)
                    dataCollector.AppendStdevFitnessSequence(
                            trackNumber,
                            "PSO_Pbest",
                            pbestStdevSequence)
                    dataCollector.AppendStdevFitnessSequence(
                            trackNumber,
                            "PSO_Episode",
                            episodeStdevSequence)
                    dataCollector.AddTimeInSecondsFromTraining(
                            trackNumber,
                            "PSO",
                            trainingTime)
                    dataCollector.AddTimeInEpisodesFromTraining(
                            trackNumber,
                            "PSO",
                            episodeCounter + 1)
                    dataCollector.AddToSearchCounter(
                            trackNumber,
                            "PSO",
                            searchCounter)
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
    train_pso(options, trainingLog)
