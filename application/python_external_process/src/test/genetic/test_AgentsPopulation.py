from ddt import ddt, data, unpack
from src.genetic.AgentsPopulation import *
import unittest

@ddt
class TestAgentsPopulation(unittest.TestCase):
	@unpack
	@data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
	def test_Constructor(self, expectedNumberOfAgents, expectedAgentsDimensions):
		population = AgentsPopulation(
				expectedNumberOfAgents,
				expectedAgentsDimensions)

		actualNumberOfAgents = len(population._agents)
		self.assertEqual(actualNumberOfAgents, expectedNumberOfAgents)
		
		actualAgentDimensions_1 = population._agentDimensions
		self.assertEqual(actualAgentDimensions_1, expectedAgentsDimensions)
		
		tempAgent = population._agents[0]
		actualAgentDimensions_2 = [tempAgent._layers[0].in_features ]
		actualAgentDimensions_2 = actualAgentDimensions_2 + \
				[tempAgent._layers[i].out_features
					for i in range(len(tempAgent._layers))]

		self.assertEqual(actualAgentDimensions_2, expectedAgentsDimensions)

	@unpack
	@data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
	def test_GetParameters(self, numberOfAgents, agentsDimensions):
		population = AgentsPopulation(numberOfAgents, agentsDimensions)
		actualParameters = population.GetParameters()
		
		expectedParametersType = torch.Tensor
		actualParametersType = type(actualParameters)
		self.assertEqual(actualParametersType, expectedParametersType)
		
		numberOfParameters = 0
		numberOfLayers = len(agentsDimensions) - 1
		for i in range(numberOfLayers):
			numberOfWeights = agentsDimensions[i] * agentsDimensions[i+1]
			numberOfBiases = agentsDimensions[i+1]
			parametersPerLayer = numberOfWeights + numberOfBiases
			numberOfParameters = numberOfParameters + parametersPerLayer
		
		expectedParametersDimensions = [numberOfAgents, numberOfParameters]
		actualParametersDimensions = actualParameters.size()
		self.assertEqual(
				list(actualParametersDimensions),
				expectedParametersDimensions)
		
		expectedParameters = []
		for agent in population._agents:
			expectedAgentParameters = []
			for layer in agent._layers:
				layerWeights = layer.weight
				numberOfWeights = layerWeights.numel()
				weightParameters = \
						torch.reshape(layerWeights, (numberOfWeights,))
				expectedAgentParameters = \
						expectedAgentParameters + weightParameters.tolist()
				expectedAgentParameters = \
						expectedAgentParameters + layer.bias.tolist()
			
			expectedParameters.append(expectedAgentParameters)
		
		expectedParameters = torch.tensor(expectedParameters)
		self.assertTrue(torch.equal(actualParameters, expectedParameters))
