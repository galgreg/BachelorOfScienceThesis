import torch

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

class AgentFitnessEvaluator:
    def __init__(self, env, brainName):
        self.env = env
        self.brainName = brainName
    
    def __call__(self, agent):
        fitness = 0
        envInfo = self.env.reset(train_mode = True)[self.brainName]

        errorCounter = 0
        stepCounter = 0
        while True:
            try:
                inputData = envInfo.vector_observations.tolist()
                episodeReward = inputData[0][-1]
                inputData = inputData[0][:-1]
                stepCounter += 1
                outputData = agent.forward(inputData)
                if episodeReward > fitness:
                    fitness = episodeReward
                if envInfo.local_done[0]:
                    break
                envInfo = self.env.step([outputData])[self.brainName]
            except IndexError:
                errorCounter += 1
                print("Index error {0}".format(errorCounter))
                if errorCounter > 1000:
                    print("Too much errors!")
                    exit()
                continue
            
            if stepCounter > 500 and fitness < 2:
                fitness -= 10.0
                break
            
        return fitness
