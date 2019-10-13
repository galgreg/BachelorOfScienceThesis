from ddt import ddt, data, unpack
from src.AgentNeuralNetwork import *
import random
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
                    [ networkDimensions[i+1], networkDimensions[i] ]
            actualWeightDimensions = actualLayer.weight.size()
            self.assertEqual(
                    list(actualWeightDimensions),
                    expectedWeightDimensions)
            
            expectedBiasDimensions = [ networkDimensions[i+1] ]
            actualBiasDimensions = actualLayer.bias.size()
            self.assertEqual(
                    list(actualBiasDimensions),
                    expectedBiasDimensions)
            
            expectedLayerDimensions = networkDimensions[i : i + 2]
            actualLayerDimensions = \
                    [actualLayer.in_features, actualLayer.out_features]
            self.assertEqual(actualLayerDimensions, expectedLayerDimensions)

    @unpack
    @data(([5, 10, 2], True), ([5, 10], False), ([5, 20, 30, 10], False))
    def test_forward(self, networkDimensions, doesRequireGrad):
        network = AgentNeuralNetwork(networkDimensions, doesRequireGrad)
        networkLayers = network._layers
        networkInput = [random.uniform(0.0, 1.0) for i in range(5)]
        
        expectedTypeOfInput = list
        actualTypeOfInput = type(networkInput)
        self.assertEqual(actualTypeOfInput, expectedTypeOfInput)
        
        dataToForward = torch.tensor(networkInput)
        
        for layer in networkLayers:
            dataToForward = F.elu(layer(dataToForward))
        
        dataToForward[0] = min((dataToForward[0] + 1), 1)
        dataToForward[1] = min(dataToForward[1], 1)

        expectedNetworkOutput = dataToForward.tolist()
        actualNetworkOutput = network.forward(networkInput)
        
        expectedTypeOfOutput = list
        actualTypeOfOutput = type(actualNetworkOutput)
        self.assertEqual(actualTypeOfOutput, expectedTypeOfOutput)
        
        self.assertEqual(actualNetworkOutput, expectedNetworkOutput)
        
        for outputValue in actualNetworkOutput:
            expectedTypeOfValue = float
            actualTypeOfValue = type(outputValue)
            self.assertEqual(actualTypeOfValue, expectedTypeOfValue)
