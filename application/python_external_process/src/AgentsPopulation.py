from src.AgentNeuralNetwork import *
import torch

class AgentsPopulation:
	def __init__(self, numberOfAgents, agentDimensions):
		self._agentDimensions = agentDimensions
		self._agents = [
				AgentNeuralNetwork(agentDimensions)
				for i in range(numberOfAgents)
		]

	def DoForward(self, inputData):
		pass
		# outputData = []				
		# for agent in self._agents:
			# agentOutput = None
			# if not agent.IsDone():
				# agentOutput = agent.forward(inputData)			
			
			# outputData.append(agentOutput)
		
		# return outputData

	# def GetPopulationParameters(self):
		# populationParameters = []
		# for agent in self._agents:
			# agentParameters = []
			# for layer in agent._layers:
				# layerWeights = layer.weight
				# numberOfWeights = layerWeights.numel()
				# weightParameters = \
						# torch.reshape(layerWeights, (numberOfWeights,))
				# agentParameters = \
						# agentParameters + weightParameters.tolist()
				# agentParameters = \
						# agentParameters + layer.bias.tolist()
			
			# populationParameters.append(agentParameters)
		
		# return torch.tensor(populationParameters)
	
	# def SetPopulationParameters(self, newParameters):
		# for agent, agentParameters in zip(self._agents, newParameters):
			# agentParameters = agentParameters.tolist()
			# for layer in agent._layers:
				# numberOfWeights = layer.weight.numel()
				# numberOfBiases = layer.bias.numel()
				
				# weightParameters = \
						# torch.tensor( agentParameters[ : numberOfWeights] )
				# weightDimensions = tuple(layer.weight.size())
				# reshapedWeightParameters = \
						# torch.reshape(weightParameters, weightDimensions)
				# layer.weight.data = reshapedWeightParameters
				# del agentParameters[ : numberOfWeights ]
				
				# biasParameters = \
						# torch.tensor( agentParameters[ : numberOfBiases] )
				# layer.bias.data = biasParameters
				# del agentParameters[ : numberOfBiases ]
