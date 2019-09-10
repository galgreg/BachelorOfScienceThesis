""" STEP 0: Import Required Packages """
from mlagents.envs import UnityEnvironment
from docopt import docopt
from src.AgentsPopulation import *
from src.training.LearningAlgorithmFactory import *
from src.training.TrainingLog import *
from src.training.TrainingResultsRepository import *

def loadConfigData(configFilePath):
    if configFilePath is None:
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

def createEnvironmentObject(options, configData):
    pass

def computeAgentDimensions(observationSize, actionSize, configData):
    if (observationSize is None) or (actionSize is None) or (configData is None):
        raise ValueError("computeAgentDimensions() error -> "
                "at least one parameter is None!")
    
    hiddenDimensions = configData["TrainingParameters"]["networkHiddenDimensions"]
    return [ observationSize ] + hiddenDimensions + [ actionSize ]

def findIndexOfBestModel(fitnessList):
    bestFitness = float("-inf")
    indexOfBestModel = 0
    
    for i in range(len(fitnessList)):
        if fitnessList[i] > bestFitness:
            bestFitness = fitnessList[i]
            indexOfBestModel = i
    
    return indexOfBestModel

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
    # --- 11 --- #
    env.close()
    trainingLog.Append("Closed Unity environment")
    # --- 12 --- #
    whichModelIsBest = findIndexOfBestModel(fitnessList)
    shouldSavePopulation = options["--save-population"]
    resultsRepository.Save(population, whichModelIsBest, shouldSavePopulation)
    
if __name__ == "__main__":
    main()
