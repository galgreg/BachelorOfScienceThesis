from src.learning_algorithms.GeneticAlgorithm import *

class LearningAlgorithmFactory:
    def __init__(self, options, configData, trainingLog = None):
        self._options = options
        self._configData = configData
        self._trainingLog = trainingLog

    def Create(self):
        learningAlgorithm = None
        if self._options["--genetic"]:
            learningAlgorithm = self._createGeneticAlgorithm()
        elif self._options["--neat"] or self._options["--ppo"]:
            raise NotImplementedError("NEAT and PPO learning algorithms are not"
                    " yet implemented!")
        
        return learningAlgorithm

    def _createGeneticAlgorithm(self):
        algorithmParameters = \
                self._configData["LearningAlgorithms"]["genetic"]
        selectionRate = algorithmParameters["selectionPercentRate"]
        probabilityThresholdToMutateChromosome = \
                algorithmParameters["minimalProbabilityToMutateChromosome"]
        probabilityThresholdToMutateGenome = \
                algorithmParameters["minimalProbabilityToMutateGenome"]
        
        learningAlgorithm = GeneticAlgorithm(
                selectionRate,
                probabilityThresholdToMutateChromosome,
                probabilityThresholdToMutateGenome)
                
        if self._trainingLog is not None:
            logMessage = \
                    "Created GeneticAlgorithm with parameters: " \
                    "selectionPercentRate = {0}, " \
                    "probabilityThresholdToMutateChromosome = {1}, " \
                    "probabilityThresholdToMutateGenome = {2}".format(
                            selectionRate,
                            probabilityThresholdToMutateChromosome,
                            probabilityThresholdToMutateGenome)
            self._trainingLog.Append(logMessage)
        
        return learningAlgorithm
