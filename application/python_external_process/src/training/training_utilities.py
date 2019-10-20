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
        inputData = envInfo.vector_observations.tolist()
        inputData = inputData[0][:-1]
        
        stepCounter = 0
        while True:
            stepCounter += 1
            outputData = agent.forward(inputData)
            envInfo = self.env.step([outputData])[self.brainName]
            inputData = envInfo.vector_observations.tolist()
            episodeReward = inputData[0][-1]
            inputData = inputData[0][:-1]

            if episodeReward > fitness:
                fitness = episodeReward
            if envInfo.local_done[0]:
                break
            if stepCounter > 1000 and fitness < 1.0:
                fitness -= 10.0
                break
            
        return fitness
