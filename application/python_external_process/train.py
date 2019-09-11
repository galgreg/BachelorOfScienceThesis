""" STEP 0: Import Required Packages """
from mlagents.envs import UnityEnvironment
from docopt import docopt
from src.AgentsPopulation import *
from src.training.LearningAlgorithmFactory import *
from src.training.TrainingLog import *
from src.training.TrainingResultsRepository import *

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

def findIndexOfBestModel(fitnessList):
    typeOfFitnessList = type(fitnessList)
    if typeOfFitnessList != list:
        raise ValueError(
                "findIndexOfBestModel() error -> fitnessList has wrong type! " \
                "type(fitnessList) = {0}".format(typeOfFitnessList))
    
    bestFitness = float("-inf")
    indexOfBestModel = 0
    
    for i in range(len(fitnessList)):
        if fitnessList[i] > bestFitness:
            bestFitness = fitnessList[i]
            indexOfBestModel = i
    
    return indexOfBestModel

def areAllAgentsDone(agentDones):
    typeOfDones = type(agentDones)
    if typeOfDones != list:
        raise ValueError(
                "areAllAgentsDone() error -> agentDones has wrong type! " \
                "type(agentDones) = {0}".format(typeOfDones))
    
    for done in agentDones:
        if type(done) != bool:
            raise ValueError("areAllAgentsDone() error -> "
                    "some agentDones elements are not bool!")
        if not done:
            return False
    
    return True

def getBestFitness(fitnessList):
    typeOfFitnessList = type(fitnessList)
    if typeOfFitnessList != list:
        raise ValueError(
                "getBestFitness() error -> fitnessList has wrong type! " \
                 "type(fitnessList) = {0}".format(typeOfFitnessList))
    
    if len(fitnessList) > 0:
        bestFitness = float("-inf")
        for currentFitness in fitnessList:
            if currentFitness > bestFitness:
                bestFitness = currentFitness
        return bestFitness
    else:
        raise ValueError("getBestFitness() error -> fitnessList is empty!")

def main():
    # --- 1 --- #
    APP_USAGE_DESCRIPTION = """
Train neural networks to drive a car on a racetrack. Racetrack must be valid Unity ML-Agents environment.

Usage:
    train.py <config-file-path> (--genetic | --neat | --ppo) [options]
    train.py -h | --help

Options:
    -v --verbose                            Run in verbose mode
    --genetic                               Use Genetic Algorithm to learn network
    --neat                                  Use NEAT algorithm to learn network
    --ppo                                   Use PPO algorithm to learn network
    --save-population                       Save population after training
    --population=<pretrained-population>    Specify path to pretrained population
    --env-path=<unity-build>                Specify path to Unity environment build
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    # --- 2 --- #
    trainingLog = TrainingLog(isVerbose = options["--verbose"])
    # --- 3 --- #
    pathToConfigFile = options["<config-file-path>"]
    CONFIG_DATA = loadConfigData(pathToConfigFile)
    trainingLog.Append("Config data has been loaded from file: {0}".format(
            pathToConfigFile))
    del pathToConfigFile
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
    if options["--env-path"].strip() == "":
        typeOfEnvInfo = "Training will execute in Unity editor.\n"
        unityEnvMessage = unityEnvMessage + typeOfEnvInfo
    else:
        typeOfEnvInfo = "Training will execute in Unity build: {0}\n".format(
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
    brainInfo = env.reset(train_mode=True)[brain_name]
    numberOfAgents = len(brainInfo.agents)
    observationSize = brainInfo.vector_observations.shape[1]
    actionSize = brain.vector_action_space_size
    trainingLog.Append(
            "Loaded from Unity environment: numberOfAgents = {0}, " \
            "observationSize = {1}, actionSize = {2}" \
                    .format(numberOfAgents, observationSize, actionSize))
    # --- 8 --- #
    agentDimensions = \
            computeAgentDimensions(
                    observationSize,
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
    
    # --- 10 - Training loop (TODO) --- #
    TRAINING_PARAMS = CONFIG_DATA["TrainingParameters"]
    MAX_EPISODES_NUMBER = TRAINING_PARAMS["maxNumberOfEpisodes"]
    MAX_STEPS_NUMBER_FOR_EPISODE = TRAINING_PARAMS["maxNumberOfStepsPerEpisode"]
    currentBestFitness = float("-inf")
    totalBestFitness = float("-inf")
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
    # --- Episode loop --- #
    for episodeCounter in range(MAX_EPISODES_NUMBER):
        fitnessList = [0 for i in range(numberOfAgents)]
        
        envInfo = env.reset(train_mode = True)[brainName]
        listOfInputData = envInfo.vector_observations.tolist()
        agentDones = envInfo.local_done
        
        # --- Step loop --- #
        for stepCounter in range(MAX_STEPS_NUMBER_FOR_EPISODE):
            listOfOutputData = population.DoForward(listOfInputData, agentDones)
            envInfo = env.step(listOfOutputData)[brainName]
            
            listOfInputData = envInfo.vector_observations.tolist()
            agentDones = envInfo.local_done
            currentRewards = envInfo.rewards
            
            for i in range(len(currentRewards)):
                fitnessList[i] += currentRewards[i]
            
            if areAllAgentsDone(agentDones):
                trainingLog.Append(
                        "Episode {0}: all agents are done after {1} steps!" \
                        .format(episodeCounter, stepCounter + 1))
                break
        
        population.Learn(fitnessList)
        currentBestFitness = getBestFitness(fitnessList) # TODO
        
        if currentBestFitness > totalBestFitness:
            totalBestFitness = currentBestFitness
            numOfEpisodesWithoutImprovement = 0
        else:
            numOfEpisodesWithoutImprovement += 1
        
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
    
    # --- 11 --- #
    env.close()
    trainingLog.Append("Closed Unity environment")
    # --- 12 --- #
    whichModelIsBest = findIndexOfBestModel(fitnessList)
    shouldSavePopulation = options["--save-population"]
    resultsRepository.Save(population, whichModelIsBest, shouldSavePopulation)
    
if __name__ == "__main__":
    main()
