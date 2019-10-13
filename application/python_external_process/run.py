from mlagents.envs import UnityEnvironment
from docopt import docopt
from src.training.TrainingResultsRepository import *

def main():
    APP_USAGE_DESCRIPTION = """
Run best trained model of car on a specified racetrack. Racetrack must be valid Unity ML-Agents environment.

Usage:
    run.py --population=<pretrained-population> [options]
    run.py -h | --help

Options:
    --population=<pretrained-population>    Specify path to pretrained population
    --env-path=<unity-build>                Specify path to Unity environment build
"""
    options = docopt(APP_USAGE_DESCRIPTION)

    env = UnityEnvironment(file_name = options["--env-path"])
    brainName = env.brain_names[0]
    brain = env.brains[brainName]
    actionSize = brain.vector_action_space_size[0]
    brainInfo = env.reset(train_mode=False)[brainName]
    del brain

    locationForPretrainedPopulation = options["--population"]
    resultsRepository = TrainingResultsRepository()
    bestAgent = resultsRepository.LoadBestModel(locationForPretrainedPopulation)
    
    try:
        while True:
            envInfo = env.reset(train_mode = False)[brainName]
            inputData = envInfo.vector_observations.tolist()
            inputData = inputData[0][:-1]
            while True:
                outputData = bestAgent.forward(inputData)
                envInfo = env.step([outputData])[brainName]
                inputData = envInfo.vector_observations.tolist()
                inputData = inputData[0][:-1]
                if envInfo.local_done[0]:
                    break
    
    except KeyboardInterrupt:
        print("Run interrupted because of KeyboardInterrupt!")
    
    print("End of run!")
    env.close()
    print("Closed Unity environment.")
    
if __name__ == "__main__":
    main()
