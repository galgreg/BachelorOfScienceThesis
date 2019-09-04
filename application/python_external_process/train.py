""" STEP 0: Import Required Packages """
from mlagents.envs import UnityEnvironment
from docopt import docopt
from src.AgentsPopulation import *

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

def load_model(pathToModel):
    # TODO
    pass

def computeAgentDimensions(observationSize, actionSize, configData):
    if (observationSize is None) or (actionSize is None) or (configData is None):
        raise ValueError("computeAgentDimensions() error -> "
                "at least one parameter is None!")
    
    hiddenDimensions = configData["TrainingParameters"]["networkHiddenDimensions"]
    return [ observationSize ] + hiddenDimensions + [ actionSize ]

def createInitialPopulation(pretrainedModel, numberOfAgents, agentDimensions):
    # TODO
    pass

def saveBestTrainedModel(population, fitnessList, locationToSave):
    # TODO
    pass

def main():
    # --- 2 --- #
    APP_USAGE_DESCRIPTION = """
Train neural networks to drive a car on a racetrack. Racetrack must be valid Unity ML-Agents environment.

Usage:
    train.py <config-file-path> (--genetic | --neat | --ppo) [options]
    train.py -h | --help

Options:
    -v --verbose                    Run in verbose mode
    --genetic                       Use Genetic Algorithm to learn network
    --neat                          Use NEAT algorithm to learn network
    --ppo                           Use PPO algorithm to learn network
    --model=<pretrained-model>      Specify path to pretrained network
    --env-path=<unity-build>        Specify path to Unity environment build
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    # --- 3 --- #
    pathToConfigFile = options["<config-file-path>"]
    CONFIG_DATA = loadConfigData(pathToConfigFile)
    # --- 4 --- #
    VERBOSE_MODE = options["--verbose"]
    # --- 5 (TODO) --- #
    pathToModel = options["--model"]
    pretrainedModel = load_model(pathToModel)
    # --- 6 --- #
    UNITY_ENV_CONFIG = CONFIG_DATA["UnityEnvironment"]
    env = UnityEnvironment(
            file_name = options["--env-path"],
            no_graphics = UNITY_ENV_CONFIG["no_graphics"],
            timeout_wait = UNITY_ENV_CONFIG["timeout_wait"],
            args = UNITY_ENV_CONFIG["args"])
    # --- 7 --- #
    brainName = env.brain_names[0]
    brain = env.brains[brainName]
    brainInfo = env.reset(train_mode=True)[brain_name]
    numberOfAgents = len(brainInfo.agents)
    observationSize = brainInfo.vector_observations.shape[1]
    actionSize = brain.vector_action_space_size
    # --- 8  --- #
    agentDimensions = \
            computeAgentDimensions(
                    observationSize,
                    actionSize,
                    CONFIG_DATA)
    # --- 9 (TODO) --- #
    population = createInitialPopulation(
            pretrainedModel,
            numberOfAgents,
            agentDimensions)
    # --- 10 - Training loop (TODO) --- #
    # --- 11 --- #
    env.close()
    # --- 12 (TODO) --- #
    saveBestTrainedModel(population, fitnessList, locationToSave)
    
if __name__ == "__main__":
    main()
