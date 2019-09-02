from ddt import ddt, data, unpack
from src.AgentsPopulation import *
import random
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

	# @unpack
	# @data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
	# def test_DoForward_AllAgentsAreNotDone(
			# self,
			# numberOfAgents,
			# agentsDimensions):
		# population = AgentsPopulation(numberOfAgents, agentsDimensions)

		# for agent in population._agents:
			# self.assertFalse(agent.IsDone())
		
		# inputData = [
				# random.uniform(0.0, 1.0)
				# for i in range(agentsDimensions[0])
		# ]
		# outputData = population.DoForward(inputData)
		
		# expectedTypeOfOutput = list
		# actualTypeOfOutput = type(outputData)
		# self.assertEqual(actualTypeOfOutput, expectedTypeOfOutput)
		
		# expectedSizeOfOutput = numberOfAgents
		# actualSizeOfOutput = len(outputData)
		# self.assertEqual(actualSizeOfOutput, expectedSizeOfOutput)
		
		# for outputValue in outputData:
			# expectedTypeOfValue = list
			# actualTypeOfValue = type(outputValue)
			# self.assertEqual(actualTypeOfValue, expectedTypeOfValue)
			
			# expectedSizeOfValue = agentsDimensions[-1]
			# actualSizeOfValue = len(outputValue)
			# self.assertEqual(actualSizeOfValue, expectedSizeOfValue)
		
	# @unpack
	# @data((20, [2, 1], [1, 2, 15, 17]), (10, [5, 3, 2], [0, 3, 6, 9]))
	# def test_DoForward_SomeAgentsAreDone(
			# self,
			# numberOfAgents,
			# agentsDimensions,
			# agentIndicesToDone):
		# population = AgentsPopulation(numberOfAgents, agentsDimensions)
		
		# for i in range(numberOfAgents):
			# if i in agentIndicesToDone:
				# population._agents[i]._done = True
			
		# for i in range(numberOfAgents):
			# if i in agentIndicesToDone:
				# self.assertTrue(population._agents[i].IsDone())
			# else:
				# self.assertFalse(population._agents[i].IsDone())
		
		# inputData = [
				# random.uniform(0.0, 1.0)
				# for i in range(agentsDimensions[0])
		# ]
		# outputData = population.DoForward(inputData)
		
		# expectedTypeOfOutput = list
		# actualTypeOfOutput = type(outputData)
		# self.assertEqual(actualTypeOfOutput, expectedTypeOfOutput)
		
		# expectedSizeOfOutput = numberOfAgents
		# actualSizeOfOutput = len(outputData)
		# self.assertEqual(actualSizeOfOutput, expectedSizeOfOutput)
		
		# for i in range(numberOfAgents):
			# if i in agentIndicesToDone:
				# self.assertTrue(outputData[i] is None)
			# else:
				# expectedTypeOfValue = list
				# actualTypeOfValue = type(outputData[i])
				# self.assertEqual(actualTypeOfValue, expectedTypeOfValue)
				
				# expectedSizeOfValue = agentsDimensions[-1]
				# actualSizeOfValue = len(outputData[i])
				# self.assertEqual(actualSizeOfValue, expectedSizeOfValue)

	# @unpack
	# @data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
	# def test_DoForward_AllAgentsAreDone(self, numberOfAgents, agentsDimensions):
		# population = AgentsPopulation(numberOfAgents, agentsDimensions)
		
		# for agent in population._agents:
			# agent._done = True
		
		# for agent in population._agents:
			# self.assertTrue(agent.IsDone())
		
		# inputData = [
				# random.uniform(0.0, 1.0)
				# for i in range(agentsDimensions[0])
		# ]
		# outputData = population.DoForward(inputData)
		
		# expectedTypeOfOutput = list
		# actualTypeOfOutput = type(outputData)
		# self.assertEqual(actualTypeOfOutput, expectedTypeOfOutput)
		
		# expectedSizeOfOutput = numberOfAgents
		# actualSizeOfOutput = len(outputData)
		# self.assertEqual(actualSizeOfOutput, expectedSizeOfOutput)
		
		# for outputValue in outputData:
			# self.assertTrue(outputValue is None)

	# @unpack
	# @data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 20, 30, 7]))
	# def test_GetPopulationParameters(self, numberOfAgents, agentsDimensions):
		# population = AgentsPopulation(numberOfAgents, agentsDimensions)
		# actualParameters = population.GetPopulationParameters()
		
		# expectedParametersType = torch.Tensor
		# actualParametersType = type(actualParameters)
		# self.assertEqual(actualParametersType, expectedParametersType)
		
		# numberOfParameters = self._computeNumberOfParameters(agentsDimensions)
		
		# expectedParametersDimensions = [numberOfAgents, numberOfParameters]
		# actualParametersDimensions = actualParameters.size()
		# self.assertEqual(
				# list(actualParametersDimensions),
				# expectedParametersDimensions)
		
		# expectedParameters = []
		# for agent in population._agents:
			# expectedAgentParameters = []
			# for layer in agent._layers:
				# layerWeights = layer.weight
				# numberOfWeights = layerWeights.numel()
				# weightParameters = \
						# torch.reshape(layerWeights, (numberOfWeights,))
				# expectedAgentParameters = \
						# expectedAgentParameters + weightParameters.tolist()
				# expectedAgentParameters = \
						# expectedAgentParameters + layer.bias.tolist()
			
			# expectedParameters.append(expectedAgentParameters)
		
		# expectedParameters = torch.tensor(expectedParameters)
		# self.assertTrue(torch.equal(actualParameters, expectedParameters))

	# @unpack
	# @data((1, [5, 2]), (10, [5, 3, 2]), (100, [5, 7, 4, 2]))
	# def test_SetPopulationParameters(self, numberOfAgents, agentsDimensions):
		# population = AgentsPopulation(numberOfAgents, agentsDimensions)
		# numberOfParameters = self._computeNumberOfParameters(agentsDimensions)
		# newParameters = torch.randn(numberOfAgents, numberOfParameters)
		
		# parametersBeforeSet = self._retrieveParametersFromPopulation(population)
		# population.SetPopulationParameters(newParameters)
		# parametersAfterSet = self._retrieveParametersFromPopulation(population)
		
		# for oldAgentParameters, newAgentParameters \
				# in zip(parametersBeforeSet, parametersAfterSet):
			# for oldLayerParameters, newLayerParameters \
					# in zip(oldAgentParameters, newAgentParameters):
				# # Assert weight and bias parameters
				# for oldParameters, newParameters \
						# in zip(oldLayerParameters, newLayerParameters):
					# expectedTypeOfParameters = torch.Tensor
					# self.assertEqual(
							# type(oldParameters),
							# expectedTypeOfParameters)
					# self.assertEqual(
							# type(newParameters),
							# expectedTypeOfParameters)
					# expectedDimensionsOfParameters = \
							# list(oldParameters.size())
					# actualDimensionsOfParameters = \
							# list(newParameters.size())
					# self.assertEqual(
							# actualDimensionsOfParameters,
							# expectedDimensionsOfParameters)				
					# self.assertFalse(
							# torch.equal(newParameters, oldParameters))

	# def _retrieveParametersFromPopulation(self, population):
		# populationParameters = []
		# for agent in population._agents:
			# agentParameters = []
			# for layer in agent._layers:
				# layerParameters = []
				# copyOfWeights = layer.weight.data.clone().detach()
				# layerParameters.append(copyOfWeights)
				# copyOfBiases = layer.bias.data.clone().detach()
				# layerParameters.append(copyOfBiases)
				# agentParameters.append(layerParameters)
			# populationParameters.append(agentParameters)
		
		# return populationParameters

	# def _computeNumberOfParameters(self, agentsDimensions):
		# numberOfParameters = 0
		# numberOfLayers = len(agentsDimensions) - 1
		# for i in range(numberOfLayers):
			# numberOfWeights = agentsDimensions[i] * agentsDimensions[i+1]
			# numberOfBiases = agentsDimensions[i+1]
			# parametersPerLayer = numberOfWeights + numberOfBiases
			# numberOfParameters = numberOfParameters + parametersPerLayer
		# return numberOfParameters
