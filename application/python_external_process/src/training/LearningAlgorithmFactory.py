class LearningAlgorithmFactory:
    def __init__(self, options, configData, trainingLog = None):
        self._options = options
        self._configData = configData
        self._trainingLog = trainingLog

    def Create(self):
        learningAlgorithm = None
        if self._options["--neat"] or self._options["--ppo"]:
            raise NotImplementedError("NEAT and PPO learning algorithms are not"
                    " yet implemented!")
        return learningAlgorithm
