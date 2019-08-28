from ddt import ddt, data, unpack
from src.AgentNeuralNetwork import *
import torch
import unittest

@ddt
class TestAgentNeuralNetwork(unittest.TestCase):
	@unpack
	@data(([5, 3, 2], True), ([3, 5, 30, 10, 2], False), ([10, 5], False))
	def test_Constructor(self, networkDimensions, doesRequireGrad):
		network = AgentNeuralNetwork(
				dimensions = networkDimensions,
				requires_grad = doesRequireGrad)
		self.assertFalse(network._done)
		networkLayers = network._layers
		expectedNumberOfLayers = len(networkDimensions) - 1
		actualNumberOfLayers = len(networkLayers)
		
		expectedRequiresGrad = doesRequireGrad
		for i, actualLayer in zip(range(expectedNumberOfLayers), networkLayers):
			actualRequiresGrad_Weight = actualLayer.weight.requires_grad
			self.assertEqual(actualRequiresGrad_Weight, expectedRequiresGrad)
			actualRequiresGrad_Bias = actualLayer.bias.requires_grad
			self.assertEqual(actualRequiresGrad_Bias, expectedRequiresGrad)
			
			expectedWeightDimensions = \
					torch.Size(networkDimensions[i+1 : i-1 : -1])
			actualWeightDimensions = actualLayer.weight.size()
			
			expectedBiasDimensions = torch.Size([ networkDimensions[i+1] ])
			actualBiasDimensions = actualLayer.bias.size()
			
			expectedLayerDimensions = networkDimensions[i : i + 2]
			actualLayerDimensions = \
					[actualLayer.in_features, actualLayer.out_features]
			self.assertEqual(actualLayerDimensions, expectedLayerDimensions)

	@unpack
	@data(([5, 10, 2], True), ([5, 10], False), ([5, 20, 30, 10], False))
	def test_forward(self, networkDimensions, doesRequireGrad):
		network = AgentNeuralNetwork(networkDimensions, doesRequireGrad)
		networkLayers = network._layers
		networkInput = torch.empty(1, 5).uniform_()
		dataToForward = networkInput.clone().detach()
		
		for layer in networkLayers:
			dataToForward = torch.sigmoid(layer(dataToForward))
		
		expectedNetworkOutput = 2*dataToForward - 1
		actualNetworkOutput = network.forward(networkInput[0].unsqueeze(0))
		self.assertTrue(torch.equal(actualNetworkOutput, expectedNetworkOutput))
		
		for outputBatch in actualNetworkOutput:
			for outputValue in outputBatch:
				self.assertTrue(outputValue >= -1.0)
				self.assertTrue(outputValue <= 1.0)

	@data(True, False)
	def test_IsDone(self, expectedDoneValue):
		network = AgentNeuralNetwork([2, 1])
		network._done = expectedDoneValue
		actualDoneValue = network.IsDone()
		self.assertEqual(actualDoneValue, expectedDoneValue)

	def test_Done(self):
		network = AgentNeuralNetwork([2, 1])
		expectedDoneValueBeforeCall = False
		actualDoneValueBeforeCall = network._done
		self.assertEqual(actualDoneValueBeforeCall, expectedDoneValueBeforeCall)
		
		network.Done()
		
		expectedDoneValueAfterCall = True
		actualDoneValueAfterCall = network._done
		self.assertEqual(actualDoneValueAfterCall, expectedDoneValueAfterCall)
		self.assertNotEqual(actualDoneValueAfterCall, actualDoneValueBeforeCall)
		
	def test_Reset(self):
		network = AgentNeuralNetwork([2, 1])
		network._done = True
		
		network.Reset()
		
		expectedDoneValueAfterCall = False
		actualDoneValueAfterCall = network._done
		self.assertEqual(actualDoneValueAfterCall, expectedDoneValueAfterCall)
