from src.genetic.AgentNeuralNetwork import *
import torch

class AgentsPopulation:
	def __init__(self, numberOfAgents, agentDimensions):
		self._agentDimensions = agentDimensions
		self._agents = [
				AgentNeuralNetwork(agentDimensions)
				for i in range(numberOfAgents)
		]

	def GetParameters(self):
		populationParameters = []
		for agent in self._agents:
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
		
	# def SetParameters(self, newParameters):
		# pass
