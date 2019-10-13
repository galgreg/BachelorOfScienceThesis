from mlagents.envs import UnityEnvironment
from docopt import docopt
from src.AgentsPopulation import *
from src.training.LearningAlgorithmFactory import *
from src.training.TrainingLog import *
from src.training.TrainingResultsRepository import *
from copy import deepcopy
import statistics


def loadConfigData(configFilePath):
    if type(configFilePath) != str:
        return {}
    configFilePath = configFilePath.strip()
    if configFilePath == "" or not configFilePath.endswith('.json'):
        return {}
    
    from os.path import isfile
    if not isfile(configFilePath):
        return {}
    
    configFileContent = ""
    with open(configFilePath, "r") as configFile:
        configFileContent = configFile.read()
    
    from json import loads
    configData = loads(configFileContent)
    return configData

def computeAgentDimensions(observationSize, actionSize, configData):
    typeOfObsSize = type(observationSize)
    typeOfActionSize = type(actionSize)
    typeOfConfigData = type(configData)
    
    if (typeOfObsSize != int) \
            or (typeOfActionSize != int) \
            or (typeOfConfigData != dict):
        raise ValueError(
                "computeAgentDimensions() error -> "
                "at least one parameter has wrong type (or is None)!\n" \
                "type(observationSize) = {0}, type(actionSize) = {1}, " \
                "type(configData) = {2}".format(
                        typeOfObsSize,
                        typeOfActionSize,
                        typeOfConfigData))
    
    hiddenDimensions = configData["TrainingParameters"]["networkHiddenDimensions"]
    return [ observationSize ] + hiddenDimensions + [ actionSize ]

def areAllAgentsDone(agentDones):
    for done in agentDones:
        if not done:
            return False
    
    return True

def main():
    # --- 1 --- #
    APP_USAGE_DESCRIPTION = """
Train neural networks to drive a car on a racetrack. Racetrack must be valid Unity ML-Agents environment.

Usage:
    train.py <config-file-path> (--neat | --ppo) [options]
    train.py -h | --help

Options:
    -v --verbose                            Run in verbose mode
    --neat                                  Use NEAT algorithm to learn network
    --ppo                                   Use PPO algorithm to learn network
    --save-population                       Save population after training
    --population=<pretrained-population>    Specify path to pretrained population
    --env-path=<unity-build>                Specify path to Unity environment build
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    # --- 2 --- #
    trainingLog = TrainingLog(isVerbose = options["--verbose"])
    trainingLog.Append("Training log has been created!")
    # --- 3 --- #
    pathToConfigFile = options["<config-file-path>"]
    CONFIG_DATA = loadConfigData(pathToConfigFile)
    trainingLog.Append("Config data has been loaded from file: {0}".format(
            pathToConfigFile))
    del pathToConfigFile
    # --- 4 --- #
    TRAINING_PARAMS = CONFIG_DATA["TrainingParameters"]
    RANDOM_SEED = TRAINING_PARAMS["randomSeed"]
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    trainingLog.Append("Random seed set to value: {0}".format(RANDOM_SEED))
    # --- 5 --- #
    factory = LearningAlgorithmFactory(options, CONFIG_DATA, trainingLog)
    learningAlgorithm = factory.Create()
    del factory
    # --- 6 --- #
    UNITY_ENV_CONFIG = CONFIG_DATA["UnityEnvironment"]
    env = UnityEnvironment(
            file_name = options["--env-path"],
            no_graphics = UNITY_ENV_CONFIG["no_graphics"],
            timeout_wait = UNITY_ENV_CONFIG["timeout_wait"],
            args = UNITY_ENV_CONFIG["args"])
    
    unityEnvMessage = "Established connection to the Unity environment. "
    if options["--env-path"] is None:
        typeOfEnvInfo = "Training will execute in Unity editor. "
        unityEnvMessage = unityEnvMessage + typeOfEnvInfo
    else:
        typeOfEnvInfo = "Training will execute in Unity build: {0}. ".format(
                options["--env-path"])
        unityEnvMessage = unityEnvMessage + typeOfEnvInfo
    
    envParametersInfo = "UnityEnvironment parameters: file_name = '{0}', " \
            "no_graphics = {1}, timeout_wait = {2}, args = {3}".format(
                    options["--env-path"],
                    UNITY_ENV_CONFIG["no_graphics"],
                    UNITY_ENV_CONFIG["timeout_wait"],
                    UNITY_ENV_CONFIG["args"])
    unityEnvMessage = unityEnvMessage + envParametersInfo
    trainingLog.Append(unityEnvMessage)
    del unityEnvMessage
    del envParametersInfo
    # --- 7 --- #
    brainName = env.brain_names[0]
    trainingLog.Append("Brain name: {0}".format(brainName))
    brain = env.brains[brainName]
    brainInfo = env.reset(train_mode=True)[brainName]
    numberOfAgents = len(brainInfo.agents)
    observationSize = brainInfo.vector_observations.shape[1]
    actionSize = brain.vector_action_space_size[0]
    trainingLog.Append(
            "Loaded from Unity environment: numberOfAgents = {0}, " \
            "observationSize = {1}, actionSize = {2}" \
                    .format(numberOfAgents, observationSize, actionSize))
    # --- 8 --- #
    agentDimensions = \
            computeAgentDimensions(
                    observationSize - 1,
                    actionSize,
                    CONFIG_DATA)
    trainingLog.Append("Computed agentDimensions: {0}".format(agentDimensions))
    # --- 9 --- #
    locationForPretrainedPopulation = options["--population"]
    population = None
    resultsRepository = TrainingResultsRepository(trainingLog)
    
    if locationForPretrainedPopulation is None:
        population = AgentsPopulation(
                numberOfAgents,
                agentDimensions,
                learningAlgorithm)
        trainingLog.Append("Created new population, with parameters: " \
                "numberOfAgents = {0}, agentDimensions = {1}, " \
                "learningAlgorithm = {2}".format(
                        numberOfAgents,
                        agentDimensions,
                        type(learningAlgorithm)))
    else:
        population = \
                resultsRepository.LoadPopulation(locationForPretrainedPopulation)
        population._sizeOfOutput = agentDimensions[-1]
        population._learningAlgorithm = learningAlgorithm
    
    # --- 10 - Training sequence --- #
    MAX_EPISODES_NUMBER = TRAINING_PARAMS["maxNumberOfEpisodes"]
    MAX_STEPS_NUMBER_FOR_EPISODE = TRAINING_PARAMS["maxNumberOfStepsPerEpisode"]
    currentBestFitness = float("-inf")
    currentMeanFitness = float("-inf")
    currentStdDevFitness = float("-inf")
    
    totalBestFitness = float("-inf")
    meanOfBestPopulation = float("-inf")
    stdDevOfBestPopulation = float("-inf")
    
    bestIndividual = None
    bestPopulation = None
    
    numOfEpisodesWithoutImprovement = 0
    
    trainingLog.Append(
            "Start training with parameters: " \
            "maxNumberOfEpisodes = {0}, maxNumberOfStepsPerEpisode = {1}, " \
            "minimalAcceptableFitness = {2}, " \
            "maxNumberOfEpisodesWithoutImprovement = {3}".format(
                    MAX_EPISODES_NUMBER,
                    MAX_STEPS_NUMBER_FOR_EPISODE,
                    TRAINING_PARAMS["minimalAcceptableFitness"],
                    TRAINING_PARAMS["maxNumberOfEpisodesWithoutImprovement"]))
    
    try:
        # --- Episode loop --- #
        for episodeCounter in range(MAX_EPISODES_NUMBER):
            fitnessList = [0] * numberOfAgents
            
            envInfo = env.reset(train_mode = True)[brainName]
            listOfInputData = envInfo.vector_observations.tolist()
            listOfInputData = \
                    [listOfInputData[i][:-1] for i in range(len(listOfInputData))]
            agentDones = envInfo.local_done
            
            # --- Step loop --- #
            for stepCounter in range(MAX_STEPS_NUMBER_FOR_EPISODE):
                listOfOutputData = population.DoForward(listOfInputData, agentDones)
                envInfo = env.step(listOfOutputData)[brainName]
                
                listOfInputData = envInfo.vector_observations.tolist()
                episodeRewards = \
                        [listOfInputData[i][-1] for i in range(len(listOfInputData))]
                listOfInputData = \
                        [listOfInputData[i][:-1] for i in range(len(listOfInputData))]
                
                agentDones = envInfo.local_done
                
                for i in range(len(episodeRewards)):
                    if episodeRewards[i] > fitnessList[i]:
                        fitnessList[i] = episodeRewards[i]
                
                if areAllAgentsDone(agentDones):
                    trainingLog.Append(
                            "Episode {0}: all agents are done after {1} steps!" \
                            .format(episodeCounter, stepCounter + 1))
                    break
                
                if stepCounter >= MAX_STEPS_NUMBER_FOR_EPISODE-1:
                    for i in range(len(agentDones)):
                        if not agentDones[i]:
                            fitnessList[i] += \
                                    TRAINING_PARAMS["penaltyForTooLongEpisode"]
                    trainingLog.Append(
                            "Interrupted {0} episode, reason: "
                            "reached max allowed steps for episode! "
                            "(maxNumberOfStepsPerEpisode = {1})".format(
                                    episodeCounter,
                                    MAX_STEPS_NUMBER_FOR_EPISODE))
            
            currentBestFitness = max(fitnessList)
            currentMeanFitness = statistics.mean(fitnessList)
            currentStdDevFitness = statistics.stdev(fitnessList)
            
            if currentBestFitness > totalBestFitness:
                totalBestFitness = currentBestFitness
                indexOfBestIndividual = fitnessList.index(currentBestFitness)
                bestIndividual = deepcopy(population._agents[indexOfBestIndividual])
                numOfEpisodesWithoutImprovement = 0
            else:
                numOfEpisodesWithoutImprovement += 1
            
            if currentMeanFitness > meanOfBestPopulation:
                meanOfBestPopulation = currentMeanFitness
                stdDevOfBestPopulation = currentStdDevFitness
                bestPopulation = deepcopy(population._agents)
            
            trainingLog.Append(
                    "Episode {0}: currentBest = {1}, currentMean = {2}, " \
                    "currentStdDev = {3} \n\t totalBestFitness = {4}, " \
                    "meanOfBestPopulation = {5}, stdDevOfBestPopulation = {6}".format(
                            episodeCounter,
                            currentBestFitness,
                            currentMeanFitness,
                            currentStdDevFitness,
                            totalBestFitness,
                            meanOfBestPopulation,
                            stdDevOfBestPopulation))
            
            if totalBestFitness >= TRAINING_PARAMS["minimalAcceptableFitness"]:
                trainingLog.Append(
                        "Training interrupted after {0} episodes, reason: " \
                        "reached minimal acceptable fitness for totalBestFitness." \
                        " (minimalAcceptableFitness = {1}, totalBestFitness = {2})" \
                        .format(
                                episodeCounter + 1,
                                TRAINING_PARAMS["minimalAcceptableFitness"],
                                totalBestFitness))
                break
            elif numOfEpisodesWithoutImprovement >= \
                    TRAINING_PARAMS["maxNumberOfEpisodesWithoutImprovement"]:
                trainingLog.Append(
                        "Training interrupted after {0} episodes, reason: " \
                        "too much episodes without improvement!" \
                        " (maxNumberOfEpisodesWithoutImprovement = {1}, " \
                        "totalBestFitness = {2}, minimalAcceptableFitness = {3})" \
                        .format(
                                episodeCounter + 1,
                                TRAINING_PARAMS["maxNumberOfEpisodesWithoutImprovement"],
                                totalBestFitness,
                                TRAINING_PARAMS["minimalAcceptableFitness"]))
                break
                
            population.Learn(fitnessList)
    
    except KeyboardInterrupt:
        trainingLog.Append("Training interrupted because of KeyboardInterrupt!")
    
    trainingLog.Append("End of training!")
    # --- 11 --- #
    env.close()
    trainingLog.Append("Closed Unity environment.")
    # --- 12 --- #
    shouldSavePopulation = options["--save-population"]
    population._agents = bestPopulation
    resultsRepository.Save(population, bestIndividual, shouldSavePopulation)
    
if __name__ == "__main__":
    main()
