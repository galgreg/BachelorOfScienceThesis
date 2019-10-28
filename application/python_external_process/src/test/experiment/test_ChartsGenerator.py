from src.experiment.ChartsGenerator import *
from src.experiment.ExperimentDataCollector import *
from shutil import rmtree
import os
import unittest

class TestChartsGenerator(unittest.TestCase):
    def setUp(self):
        self._dataCollector = ExperimentDataCollector()
        self._basePathForCharts = "TEST_BASE_PATH_FOR_CHARTS"
        
        if os.path.isdir(self._basePathForCharts):
            rmtree(self._basePathForCharts)
        os.mkdir(self._basePathForCharts)
        
        self._generator = \
                ChartsGenerator(self._dataCollector, self._basePathForCharts)
        
        self.pathToChartsDir = os.path.join(self._basePathForCharts, "charts")
    
    def tearDown(self):
        del self._generator
        if os.path.isdir(self._basePathForCharts):
            rmtree(self._basePathForCharts)
        del self._basePathForCharts
        del self._dataCollector

    def test_Constructor_OK(self):
        self.assertEqual(self._dataCollector, self._generator._dataCollector)
        self.assertEqual(
                self._basePathForCharts,
                self._generator._basePathForCharts)
        
        doesChartsDirExist = os.path.isdir(self.pathToChartsDir)
        self.assertTrue(doesChartsDirExist)
    
    def test_Constructor_BaseDirDoesNotExist(self):
        pathToNonExistentDir = "NON_EXISTENT_PATH"
        self.assertRaises(
                NotADirectoryError,
                ChartsGenerator,
                self._dataCollector,
                pathToNonExistentDir)

    def test_CreateMeanTrainingTimeCharts(self):
        fullFilledMeanTrainingTimesAsSeconds = {
            "RaceTrack_1" : {
                "DE": {
                    "Sum" : 600,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 540,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 800,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 765,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 999,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 987,
                    "TrialCounter" : 10
                }
            }
        }
        fullFilledMeanTrainingTimesAsEpisodes = {
            "RaceTrack_1" : {
                "DE": {
                    "Sum" : 60,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 54,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 80,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 76,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 99,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 98,
                    "TrialCounter" : 10
                }
            }
        }
        self._generator._dataCollector.MeanTrainingTimesAsSeconds = \
                fullFilledMeanTrainingTimesAsSeconds
        self._generator._dataCollector.MeanTrainingTimesAsEpisodes = \
                fullFilledMeanTrainingTimesAsEpisodes
        self._generator.CreateMeanTrainingTimeCharts()
        
        pathToChartFile_Seconds = \
                os.path.join(self.pathToChartsDir, "train_time_seconds.svg")
        doesSecondsFileExist = os.path.isfile(pathToChartFile_Seconds)
        self.assertTrue(doesSecondsFileExist)
        
        pathToChartFile_Episodes = \
                os.path.join(self.pathToChartsDir, "train_time_episodes.svg")
        doesEpisodesFileExist = os.path.isfile(pathToChartFile_Episodes)
        self.assertTrue(doesEpisodesFileExist)
        

    def test_CreateValidationCharts(self):
        fulfilledMatrices = {
            "PSO" : [[10, 8, 4], [10, 10, 7], [10, 10, 10]],
            "DE" : [[10, 7, 5], [9, 10, 4], [10, 10, 10]],
        }
        self._generator._dataCollector.ValidationMatrices = fulfilledMatrices
        self._generator.CreateValidationCharts()
        
        pathToChartFile_DE = \
                os.path.join(self.pathToChartsDir, "validation_de.svg")
        doesValidationDeExist = os.path.isfile(pathToChartFile_DE)
        self.assertTrue(doesValidationDeExist)
        
        pathToChartFile_PSO = \
                os.path.join(self.pathToChartsDir, "validation_pso.svg")
        doesValidationPsoExist = os.path.isfile(pathToChartFile_DE)
        self.assertTrue(doesValidationPsoExist)

    def test_CreateMeanSearchCounterChart(self):
        fulfilledMeanSearchCounters = {
            "RaceTrack_1" : {
                "DE": {
                    "Sum" : 600,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 540,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 800,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 765,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 999,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 987,
                    "TrialCounter" : 10
                }
            }
        }
        self._generator._dataCollector.MeanSearchCounters = \
                fulfilledMeanSearchCounters
        self._generator.CreateMeanSearchCounterChart()
        
        pathToChartFile = \
                os.path.join(self.pathToChartsDir, "search_count.svg")
        doesChartFileExist = os.path.isfile(pathToChartFile)
        self.assertTrue(doesChartFileExist)
        
