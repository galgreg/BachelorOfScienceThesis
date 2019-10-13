from mlagents.envs import UnityEnvironment
from docopt import docopt
from src.AgentsPopulation import *
from src.training.TrainingLog import *
from src.training.TrainingResultsRepository import *
import statistics
import random

def computeNumOfParameters(agentDimensions):
    numOfParameters = 0
    for i in range(len(agentDimensions) - 1):
        numOfParameters += \
                agentDimensions[i] * agentDimensions[i+1] + agentDimensions[i+1]

    return numOfParameters

def retrieveParametersFromAgentList(agentList):
    populationParameters = []
    for agent in agentList:
        agentParameters = []
        for layer in agent._layers:
            layerWeights = layer.weight
            numberOfWeights = layerWeights.numel()
            weightParameters = \
                    torch.reshape(layerWeights, (numberOfWeights,))
            agentParameters = \
                    agentParameters + weightParameters.tolist()
            agentParameters = \
                    agentParameters + layer.bias.tolist()
        
        populationParameters.append(agentParameters)

    return torch.tensor(populationParameters)

def setNewParametersOnAgent(agent, newParameters):
    agentParameters = newParameters.tolist()
    for layer in agent._layers:
        numberOfWeights = layer.weight.numel()
        numberOfBiases = layer.bias.numel()
        
        weightParameters = \
                torch.tensor( agentParameters[ : numberOfWeights] )
        weightDimensions = tuple(layer.weight.size())
        reshapedWeightParameters = \
                torch.reshape(weightParameters, weightDimensions)
        layer.weight.data = reshapedWeightParameters
        del agentParameters[ : numberOfWeights ]
        
        biasParameters = \
                torch.tensor( agentParameters[ : numberOfBiases] )
        layer.bias.data = biasParameters
        del agentParameters[ : numberOfBiases ]

def evaluateAgentFitness(env, brainName, agent, trainingLog):
    fitness = float("-inf")
    envInfo = env.reset(train_mode = True)[brainName]
    inputData = envInfo.vector_observations.tolist()
    inputData = inputData[0][:-1]
    
    stepCounter = 0
    while True:
        stepCounter += 1
        outputData = agent.forward(inputData)
        envInfo = env.step([outputData ])[brainName]
        inputData = envInfo.vector_observations.tolist()
        episodeReward = inputData[0][-1]
        inputData = inputData[0][:-1]

        if episodeReward > fitness:
            fitness = episodeReward
        if envInfo.local_done[0]:
            break
        if stepCounter > 200 and fitness < 1.0:
            fitness -= 10.0
            break
        
    return fitness


def main():
    # --- 1 - Specify script's usage options --- #
    APP_USAGE_DESCRIPTION = """
Train neural networks to drive a car on a racetrack. Racetrack must be valid Unity ML-Agents environment.

Usage:
    train_de.py [options]
    train_de.py -h | --help

Options:
    -v --verbose                            Run in verbose mode
    --save-population                       Save population after training
    --population=<pretrained-population>    Specify path to pretrained population
    --random-seed=<random_seed>             Specify random seed
    --max-episodes-number=<n>               Specify max number of episodes for training [default: 1000]
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    
    # --- 2 - Create logging object --- #
    trainingLog = TrainingLog(isVerbose = options["--verbose"])
    trainingLog.Append("Training log has been created!")
    
    # --- 3 - Set random seed --- #
    RANDOM_SEED = options["--random-seed"]
    if isinstance(RANDOM_SEED, str) and RANDOM_SEED.isdigit():
        RANDOM_SEED = int(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        trainingLog.Append("Random seed set to value: {0}".format(RANDOM_SEED))
    
    # --- 4 - Establish connection with Unity environment --- #
    env = UnityEnvironment()
    trainingLog.Append("Established connection to the Unity environment!")
   
    # --- 5 - Get info from Unity environment --- #
    brainName = env.brain_names[0]
    trainingLog.Append("Brain name: {0}".format(brainName))
    brain = env.brains[brainName]
    brainInfo = env.reset(train_mode=True)[brainName]
    observationSize = brainInfo.vector_observations.shape[1]
    actionSize = brain.vector_action_space_size[0]
    trainingLog.Append(
            "Loaded from Unity environment: observationSize = {0}, " \
            "actionSize = {1}".format(observationSize, actionSize))
    
    # --- 6 - Compute agent dimensions -- #
    HIDDEN_DIMENSIONS = [5 ]
    agentDimensions = [observationSize - 1] + HIDDEN_DIMENSIONS + [actionSize]
    trainingLog.Append("Computed agentDimensions: {0}".format(agentDimensions))
    
    # --- 7 - Create population ---- #
    locationForPretrainedPopulation = options["--population"]
    NUM_OF_AGENTS = 50
    population = None
    resultsRepository = TrainingResultsRepository(trainingLog)
    
    if locationForPretrainedPopulation is None:
        population = AgentsPopulation(NUM_OF_AGENTS, agentDimensions, None)
        trainingLog.Append("Created new population, with parameters: " \
                "NUM_OF_AGENTS = {0}, agentDimensions = {1}, ".format(
                        NUM_OF_AGENTS,
                        agentDimensions))
    else:
        population = \
                resultsRepository.LoadPopulation(locationForPretrainedPopulation)
        population._sizeOfOutput = agentDimensions[-1]
        population._learningAlgorithm = None
    
    # --- 8 - Training sequence --- #
    MAX_EPISODES_NUMBER = options["--max-episodes-number"]
    currentBestFitness = float("-inf")
    currentMeanFitness = float("-inf")
    currentStdDevFitness = float("-inf")
    
    trainingLog.Append(
            "Start training with parameters: maxNumberOfEpisodes = {0}".format(
                    MAX_EPISODES_NUMBER))
    indexOfBestAgent = -1
    
    try:
        ########### Start of Differential Evolution ############################
        MUTATION_FACTOR = 0.8
        CROSS_PROBABILITY = 0.7
        NUM_OF_PARAMS = computeNumOfParameters(agentDimensions)
        MINIMAL_ACCEPTABLE_FITNESS = 104.0 # RaceTrack_1
        # MINIMAL_ACCEPTABLE_FITNESS = 110.0 # RaceTrack_2
        # MINIMAL_ACCEPTABLE_FITNESS = 130.0 # RaceTrack_3
        
        pop_denorm = retrieveParametersFromAgentList(population._agents)
        pop_norm = pop_denorm / 4 + 0.5
        fitnessList = []
        # compute fitness for whole population - Begin #########################
        for agentIndex in range(NUM_OF_AGENTS):
            agentFitness = evaluateAgentFitness(
                    env,
                    brainName,
                    population._agents[agentIndex],
                    trainingLog)
            fitnessList.append(agentFitness)
        ########################################################################
        indexOfBestAgent = fitnessList.index(max(fitnessList))
        bestFitness = max(fitnessList)
        meanFitness = statistics.mean(fitnessList)
        stdDevFitness = statistics.stdev(fitnessList)
        
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
                
                # compute fitness for trial - Begin ###########################
                setNewParametersOnAgent(population._agents[j], trial_denorm)
                agentIndex = j
                fitness_trial = evaluateAgentFitness(
                        env,
                        brainName,
                        population._agents[agentIndex],
                        trainingLog)
                ###############################################################
                
                if fitness_trial > fitnessList[j]:
                    fitnessList[j] = fitness_trial
                    pop_denorm[j] = trial_denorm
                    pop_norm[j] = trial_norm
                    if fitness_trial > bestFitness:
                        indexOfBestAgent = j
                        bestFitness = fitness_trial
                else:
                    setNewParametersOnAgent(population._agents[j], pop_denorm[j])
                
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
        
    except KeyboardInterrupt:
        trainingLog.Append("Training interrupted because of KeyboardInterrupt!")
    
    trainingLog.Append("End of training!")
    
    # --- 9 - Close environment --- #
    env.close()
    trainingLog.Append("Closed Unity environment.")
    
    # --- 10 - Save training results --- #
    shouldSavePopulation = options["--save-population"]
    bestAgent = population._agents[indexOfBestAgent]
    resultsRepository.Save(population, bestAgent, shouldSavePopulation)
    
if __name__ == "__main__":
    main()
