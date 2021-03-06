import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNeuralNetwork(nn.Module):
    # Define topology of neural network
    def __init__(self, dimensions, requires_grad = False):
        super(AgentNeuralNetwork, self).__init__()
        self._layers = []
        for i in range(len(dimensions) - 1):
            self._layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
            self._layers[i].weight = nn.Parameter(
                    data = torch.FloatTensor(dimensions[i+1], dimensions[i]).uniform_(-2.0, 2.0),
                    requires_grad = requires_grad)
            self._layers[i].bias = nn.Parameter(
                    data = torch.FloatTensor(dimensions[i+1]).uniform_(-2.0, 2.0),
                    requires_grad = requires_grad)

    # Define how output is computed
    #   -> dataToProcess has 'list' type
    def forward(self, dataToProcess):
        dataToProcess = torch.tensor(dataToProcess)
        for networkLayer in self._layers:
            dataToProcess = F.elu(networkLayer(dataToProcess))
        dataToProcess[0] = min((dataToProcess[0] + 1), 1)
        dataToProcess[1] = min(dataToProcess[1], 1)
        return dataToProcess.tolist()
