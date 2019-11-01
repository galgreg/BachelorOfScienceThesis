from ddt import ddt, data, unpack
from src.experiment.ExperimentDataCollector import *
import unittest
import random

@ddt
class TestExperimentDataCollector(unittest.TestCase):
    def setUp(self):
        self._dataCollector = ExperimentDataCollector()
    
    def tearDown(self):
        del self._dataCollector
    
    def test_Constructor(self):
        self.assertTrue(self._dataCollector.PathToLastSavedModel is None)
        
        expectedBestFitness = {
            "RaceTrack_1" : {
                "DE": [],
                "PSO" : []
            },
            "RaceTrack_2" : {
                "DE": [],
                "PSO" : []
            },
            "RaceTrack_3" : {
                "DE": [],
                "PSO" : []
            }
        }
        self.assertEqual(expectedBestFitness, self._dataCollector.BestFitness)
        
        expectedMeanFitness = {
            "RaceTrack_1" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_2" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_3" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            }
        }
        self.assertEqual(expectedMeanFitness, self._dataCollector.MeanFitness)
        
        expectedStdevFitness = {
            "RaceTrack_1" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_2" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_3" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            }
        }
        self.assertEqual(expectedStdevFitness, self._dataCollector.StdevFitness)
        
        expectedMeanTrainingTimesAsSeconds = {
            "RaceTrack_1" : {
                "DE": {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                }
            }
        }
        self.assertEqual(
                expectedMeanTrainingTimesAsSeconds,
                self._dataCollector.MeanTrainingTimesAsSeconds)
        
        expectedMeanTrainingTimesAsEpisodes = {
            "RaceTrack_1" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            }
        }
        self.assertEqual(
                expectedMeanTrainingTimesAsEpisodes,
                self._dataCollector.MeanTrainingTimesAsEpisodes)
        
        expectedValidationMatrices = {
            "PSO" : [[0] * 3] * 3,
            "DE" : [[0] * 3] * 3
        }
        self.assertEqual(
                expectedValidationMatrices,
                self._dataCollector.ValidationMatrices)
        
        expectedMeanSearchCounters = {
            "RaceTrack_1" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            }
        }
        self.assertEqual(
                expectedMeanSearchCounters,
                self._dataCollector.MeanSearchCounters)

    @unpack
    @data((1, "PSO"), (2, "DE"), (2, "PSO"), (3, "DE"))
    def test_AppendBestFitnessSequence(self, trackNum, algorithm):
        fitnessSequence = [random.random() for _ in range(random.randrange(10))]
        
        trackName = "RaceTrack_{0}".format(trackNum)
        expectedBestFitness = {
            "RaceTrack_1" : {
                "DE": [],
                "PSO" : []
            },
            "RaceTrack_2" : {
                "DE": [],
                "PSO" : []
            },
            "RaceTrack_3" : {
                "DE": [],
                "PSO" : []
            }
        }
        expectedBestFitness[trackName][algorithm].append(fitnessSequence)
        self._dataCollector.AppendBestFitnessSequence(
                trackNum,
                algorithm,
                fitnessSequence)
        self.assertEqual(expectedBestFitness, self._dataCollector.BestFitness)
    
    @unpack
    @data((1, "PSO_Pbest"), (2, "DE"), (2, "PSO_Pbest"), (3, "PSO_Episode"))
    def test_AppendMeanFitnessSequence(self, trackNum, algorithm):
        meanSequence = [random.random() for _ in range(random.randrange(10))]
        
        trackName = "RaceTrack_{0}".format(trackNum)
        expectedMeanFitness = {
            "RaceTrack_1" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_2" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_3" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            }
        }
        expectedMeanFitness[trackName][algorithm].append(meanSequence)
        self._dataCollector.AppendMeanFitnessSequence(
                trackNum,
                algorithm,
                meanSequence)
        self.assertEqual(expectedMeanFitness, self._dataCollector.MeanFitness)
    
    @unpack
    @data((1, "PSO_Pbest"), (2, "DE"), (2, "PSO_Pbest"), (3, "PSO_Episode"))
    def test_AppendStdevFitnessSequence(self, trackNum, algorithm):
        stdevSequence = [random.random() for _ in range(random.randrange(10))]
        
        trackName = "RaceTrack_{0}".format(trackNum)
        expectedStdevFitness = {
            "RaceTrack_1" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_2" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            },
            "RaceTrack_3" : {
                "DE": [],
                "PSO_Pbest" : [],
                "PSO_Episode" : [],
            }
        }
        expectedStdevFitness[trackName][algorithm].append(stdevSequence)
        self._dataCollector.AppendStdevFitnessSequence(
                trackNum,
                algorithm,
                stdevSequence)
        self.assertEqual(expectedStdevFitness, self._dataCollector.StdevFitness)
    
    @unpack
    @data((1, "PSO"), (2, "DE"), (2, "PSO"), (3, "DE"))
    def test_AddTimeInSecondsFromTraining(self, trackNum, algorithm):
        timeInSeconds = random.uniform(2, 10)
        expectedMeanTrainingTimesAsSeconds = {
            "RaceTrack_1" : {
                "DE": {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0.0,
                    "TrialCounter" : 0
                }
            }
        }
        trackName = "RaceTrack_{0}".format(trackNum)
        expectedMeanTrainingTimesAsSeconds[trackName][algorithm]["Sum"] += timeInSeconds
        expectedMeanTrainingTimesAsSeconds[trackName][algorithm]["TrialCounter"] += 1
        self._dataCollector.AddTimeInSecondsFromTraining(
                trackNum,
                algorithm,
                timeInSeconds)
        self.assertEqual(
                expectedMeanTrainingTimesAsSeconds,
                self._dataCollector.MeanTrainingTimesAsSeconds)

    @unpack
    @data((1, "PSO"), (2, "DE"), (2, "PSO"), (3, "DE"))
    def test_AddTimeInEpisodesFromTraining(self, trackNum, algorithm):
        timeInEpisodes = random.randrange(2, 10)
        expectedMeanTrainingTimesAsEpisodes = {
            "RaceTrack_1" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            }
        }
        trackName = "RaceTrack_{0}".format(trackNum)
        expectedMeanTrainingTimesAsEpisodes[trackName][algorithm]["Sum"] += timeInEpisodes
        expectedMeanTrainingTimesAsEpisodes[trackName][algorithm]["TrialCounter"] += 1
        self._dataCollector.AddTimeInEpisodesFromTraining(
                trackNum,
                algorithm,
                timeInEpisodes)
        self.assertEqual(
                expectedMeanTrainingTimesAsEpisodes,
                self._dataCollector.MeanTrainingTimesAsEpisodes)
    
    @unpack
    @data(("PSO", 1, 2), ("PSO", 2, 1), ("DE", 1, 3), ("DE", 3, 1))
    def test_IncrementValidationMatrixEntry(self, algorithm, trainNum, runNum):
        numOfIncrements = random.randrange(2, 10)
        for _ in range(numOfIncrements):
            self._dataCollector.IncrementValidationMatrixEntry(
                    algorithm,
                    trainNum,
                    runNum)
        
        expectedValidationMatrices = {
            "PSO" : [[0] * 3] * 3,
            "DE" : [[0] * 3] * 3
        }
        expectedValidationMatrices[algorithm][trainNum-1][runNum-1] = \
                numOfIncrements
        self.assertEqual(
                expectedValidationMatrices,
                self._dataCollector.ValidationMatrices)

    @unpack
    @data((1, "PSO"), (2, "DE"), (2, "PSO"), (3, "DE"))
    def test_AddToSearchCounter(self, trackNum, algorithm):
        numOfSearches = random.randrange(30, 1000)
        numOfTimesToAdd = random.randrange(2, 5)
        
        for _ in range(numOfTimesToAdd):
            self._dataCollector.AddToSearchCounter(
                    trackNum,
                    algorithm,
                    numOfSearches)
        
        expectedMeanSearchCounters = {
            "RaceTrack_1" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 0,
                    "TrialCounter" : 0
                },
                "PSO" : {
                    "Sum" : 0,
                    "TrialCounter" : 0
                }
            }
        }
        trackName = "RaceTrack_{0}".format(trackNum)
        expectedMeanSearchCounters[trackName][algorithm]["Sum"] = \
                numOfSearches * numOfTimesToAdd
        expectedMeanSearchCounters[trackName][algorithm]["TrialCounter"] = \
                numOfTimesToAdd
        
        self.assertEqual(
                expectedMeanSearchCounters,
                self._dataCollector.MeanSearchCounters)
        
