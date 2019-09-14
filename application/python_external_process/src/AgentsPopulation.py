from src.AgentNeuralNetwork import *
import torch

class AgentsPopulation:
    def __init__(self, numberOfAgents, agentDimensions, learningAlgorithm):
        self._sizeOfOutput = agentDimensions[-1]
        self._learningAlgorithm = learningAlgorithm
        self._agents = [
                AgentNeuralNetwork(agentDimensions)
                for i in range(numberOfAgents)
        ]

    def DoForward(self, listOfInputData, agentDones):
        listOfOutputData = []               
        for agent, inputData, isDone \
                in zip(self._agents, listOfInputData, agentDones):
            agentOutput = [0.0 for i in range(self._sizeOfOutput)]
            if not isDone:
                agentOutput = agent.forward(inputData)
            listOfOutputData.append(agentOutput)
        
        return listOfOutputData

    def Learn(self, rewardList):
        newAgents = \
                self._learningAlgorithm.ComputeNewPopulation(
                        self._agents,
                        rewardList)
        self._agents = newAgents
