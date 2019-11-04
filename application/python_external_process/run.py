from mlagents.envs import UnityEnvironment
from docopt import docopt
from src.training.TrainingResultsRepository import *
from src.Logger import *

def getProgramOptions():
    APP_USAGE_DESCRIPTION = """
Run best trained model of car on a specified racetrack. Racetrack must be valid Unity ML-Agents environment.

Usage:
    run.py --model=<pretrained-model> [options]
    run.py -h | --help

Options:
    --model=<pretrained-model>      Specify path to pretrained model
    --env-path=<unity-build>        Specify path to Unity environment build
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    return options

def run(options, runLog, minimumAcceptableFitness = None):   
    isFunctionInValidationMode = \
            isinstance(minimumAcceptableFitness, float)
    runLog.Append("This is run.py -> script for running pretrained models!")
    
    locationOfPretrainedModel = options["--model"]
    
    resultsRepository = TrainingResultsRepository()
    bestAgent = resultsRepository.LoadBestModel(locationOfPretrainedModel)
    
    if bestAgent is None:
        runLog.Append("run.run() error: Cannot load model, location " \
                "'training_results/{0}' does not exist!".format(
                        locationOfPretrainedModel))
        exit()
    
    runLog.Append("Run model from 'training_results/{0}'!".format(
            locationOfPretrainedModel))

    pathToEnv = options["--env-path"]
    env = UnityEnvironment(file_name = pathToEnv)
    if pathToEnv is None:
        runLog.Append("Established connection with Unity Editor!")
    else:
        runLog.Append("Established connection with Unity build '{0}'!" \
                .format(pathToEnv))
    del pathToEnv

    brainName = env.brain_names[0]

    if isFunctionInValidationMode:
        fitness = 0.0

    shouldRunBeExecuted = True
    try:
        while shouldRunBeExecuted:
            envInfo = env.reset(train_mode = False)[brainName]
            inputData = envInfo.vector_observations.tolist()
            inputData = inputData[0][0:-1]
            while shouldRunBeExecuted:
                outputData = bestAgent.forward(inputData)
                envInfo = env.step([outputData])[brainName]
                inputData = envInfo.vector_observations.tolist()
                
                if isFunctionInValidationMode:
                    episodeReward = inputData[0][-1]
                    if episodeReward > fitness:
                        fitness = episodeReward

                inputData = inputData[0][:-1]

                if envInfo.local_done[0]:
                    if isFunctionInValidationMode:
                        shouldRunBeExecuted = False
                    break
    
    except KeyboardInterrupt:
        runLog.Append("\nRun interrupted because of KeyboardInterrupt!")
    
    runLog.Append("End of run!")
    env.close()
    runLog.Append("Closed Unity environment.")
    
    if isFunctionInValidationMode:
        return fitness >= minimumAcceptableFitness
    else:
        return False
    
if __name__ == "__main__":
    options = getProgramOptions()
    runLog = Logger(isVerbose = True, fileName = "run")
    run(options, runLog)
