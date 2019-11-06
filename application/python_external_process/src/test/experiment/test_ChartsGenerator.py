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

    def _assertFileContent(self, pathToFile, expectedFileContent):
        doesFileExist = os.path.isfile(pathToFile)
        self.assertTrue(doesFileExist)
        
        with open(pathToFile, "r") as actualFile:
            actualFileContent = actualFile.read()
            self.assertEqual(actualFileContent, expectedFileContent)

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

    def test_ExportBestComparisonToCsv(self):
        fullFilledBestFitness = {
            "RaceTrack_1" : {
                "DE": [ [2.14, 3.57, 14.03],
                    [2.34, 3.7, 5.13, 14.11],
                    [3.31, 15.03] ],
                "PSO" : [ [1.95, 4.21, 5.13, 14.99],
                    [1.74, 8.13, 14.56],
                    [1.23, 5.67, 9.99, 14.0] ]
            },
            "RaceTrack_2" : {
                "DE": [ [2.13, 7.57, 13.07],
                    [2.73, 7.7, 5.17, 13.11],
                    [7.71, 15.07],
                    [3.14, 5.68, 9.46, 14.13]],
                "PSO" : [ [1.95, 3.21, 5.17, 13.99],
                    [1.73, 8.17, 13.56],
                    [1.27, 5.67, 9.99, 13.0],
                    [2.14, 6.8, 9.7, 15.3]]
            },
            "RaceTrack_3" : {
                "DE": [ [2.14, 3.57, 4.13, 5.11, 6.18, 6.48, 6.7, 8.13, 9.27, \
                        11.35, 12.48, 13.57, 14.03, 14.35, 15.78, 17.33, 19.21],
                    [2.73, 7.7, 5.17, 13.11]],
                "PSO" : [ [1.95, 4.21, 5.13, 14.99, 20.0, 20.5, 21.3, 22.4, \
                        23.1, 24.2, 24.6, 25.6, 26.11],
                    [7.71, 15.07] ]
            }
        }
        expectedFileContents = [
            "DE,2.14,3.57,14.03\nPSO,1.95,4.21,5.13,14.99\n",
            "DE,2.34,3.7,5.13,14.11\nPSO,1.74,8.13,14.56\n",
            "DE,3.31,15.03\nPSO,1.23,5.67,9.99,14.0\n",
            "DE,2.13,7.57,13.07\nPSO,1.95,3.21,5.17,13.99\n",
            "DE,2.73,7.7,5.17,13.11\nPSO,1.73,8.17,13.56\n",
            "DE,7.71,15.07\nPSO,1.27,5.67,9.99,13.0\n",
            "DE,3.14,5.68,9.46,14.13\nPSO,2.14,6.8,9.7,15.3\n",
            "DE,2.14,3.57,4.13,5.11,6.18,6.48,6.7,8.13,9.27," \
            "11.35,12.48,13.57,14.03,14.35,15.78,17.33,19.21\n" \
            "PSO,1.95,4.21,5.13,14.99,20.0,20.5,21.3,22.4,23.1,24.2," \
            "24.6,25.6,26.11\n",
            "DE,2.73,7.7,5.17,13.11\nPSO,7.71,15.07\n"
        ]
        self._generator._dataCollector.BestFitness = fullFilledBestFitness
        self._generator.ExportBestComparisonToCsv()
        
        expectedListOfCsvFiles = [
            "best_track_1_trial_01.csv",
            "best_track_1_trial_02.csv",
            "best_track_1_trial_03.csv",
            "best_track_2_trial_01.csv",
            "best_track_2_trial_02.csv",
            "best_track_2_trial_03.csv",
            "best_track_2_trial_04.csv",
            "best_track_3_trial_01.csv",
            "best_track_3_trial_02.csv"
        ]
        
        for fileName, expectedFileContent \
                in zip(expectedListOfCsvFiles, expectedFileContents):
            fullPathToFile = os.path.join(self.pathToChartsDir, fileName)
            self._assertFileContent(fullPathToFile, expectedFileContent)

    def test_ExportMeanComparisonToCsv(self):
        fullFilledMeanFitness = {
            "RaceTrack_1" : {
                "DE": [ [2.14, 3.57, 14.03],
                    [2.34, 3.7, 5.13, 14.11],
                    [3.31, 15.03] ],
                "PSO_Pbest" : [ [1.95, 4.21, 5.13, 14.99],
                    [1.74, 8.13, 14.56],
                    [1.23, 5.67, 9.99, 14.0] ],
                "PSO_Episode" : [ [1.34, 3.45, 4.91, 12.33],
                    [1.5, 7.33, 13.78],
                    [0.99, 4.21, 8.88, 13.5] ]
            },
            "RaceTrack_2" : {
                "DE": [ [2.13, 7.57, 13.07],
                    [2.73, 7.7, 5.17, 13.11],
                    [7.71, 15.07],
                    [3.14, 5.68, 9.46, 14.13]],
                "PSO_Pbest" : [ [1.95, 3.21, 5.17, 13.99],
                    [1.73, 8.17, 13.56],
                    [1.27, 5.67, 9.99, 13.0],
                    [2.14, 6.8, 9.7, 15.3]],
                "PSO_Episode" : [ [1.95, 2.21, 4.17, 12.79],
                    [1.23, 5.67, 9.99],
                    [0.81, 4.21, 8.51, 12.1],
                    [1.23, 5.67, 7.7, 11.1]],
            },
            "RaceTrack_3" : {
                "DE": [ [2.14, 3.57, 4.13, 5.11, 6.18, 6.48, 6.7, 8.13, 9.27, \
                        11.35, 12.48, 13.57, 14.03, 14.35, 15.78, 17.33, 19.21],
                    [2.73, 7.7, 5.17, 13.11]],
                "PSO_Pbest" : [ [1.95, 4.21, 5.13, 14.99, 20.0, 20.5, 21.3, \
                        22.4, 23.1, 24.2, 24.6, 25.6, 26.11],
                    [7.71, 15.07] ],
                "PSO_Episode" : [ [1.35, 3.21, 4.13, 12.99, 19.0, 19.5, 20.3, \
                        21.4, 22.1, 23.2, 23.6, 24.6, 25.11],
                    [5.71, 13.07] ]
            }
        }
        self._generator._dataCollector.MeanFitness = fullFilledMeanFitness
        self._generator.ExportMeanComparisonToCsv()
        
        expectedFileContents = [
            "DE,2.14,3.57,14.03\nPSO_Pbest,1.95,4.21,5.13,14.99\n" \
                "PSO_Episode,1.34,3.45,4.91,12.33\n",
            "DE,2.34,3.7,5.13,14.11\nPSO_Pbest,1.74,8.13,14.56\n" \
                "PSO_Episode,1.5,7.33,13.78\n",
            "DE,3.31,15.03\nPSO_Pbest,1.23,5.67,9.99,14.0\n" \
                "PSO_Episode,0.99,4.21,8.88,13.5\n",
            "DE,2.13,7.57,13.07\nPSO_Pbest,1.95,3.21,5.17,13.99\n" \
                "PSO_Episode,1.95,2.21,4.17,12.79\n",
            "DE,2.73,7.7,5.17,13.11\nPSO_Pbest,1.73,8.17,13.56\n" \
                "PSO_Episode,1.23,5.67,9.99\n",
            "DE,7.71,15.07\nPSO_Pbest,1.27,5.67,9.99,13.0\n" \
                "PSO_Episode,0.81,4.21,8.51,12.1\n",
            "DE,3.14,5.68,9.46,14.13\nPSO_Pbest,2.14,6.8,9.7,15.3\n" \
                "PSO_Episode,1.23,5.67,7.7,11.1\n",
            "DE,2.14,3.57,4.13,5.11,6.18,6.48,6.7,8.13,9.27," \
                "11.35,12.48,13.57,14.03,14.35,15.78,17.33,19.21\n" \
                "PSO_Pbest,1.95,4.21,5.13,14.99,20.0,20.5,21.3,22.4,23.1,24.2," \
                "24.6,25.6,26.11\nPSO_Episode,1.35,3.21,4.13,12.99,19.0,19.5," \
                "20.3,21.4,22.1,23.2,23.6,24.6,25.11\n",
            "DE,2.73,7.7,5.17,13.11\nPSO_Pbest,7.71,15.07\n" \
                "PSO_Episode,5.71,13.07\n"
        ]
        
        expectedListOfCsvFiles = [
            "mean_track_1_trial_01.csv",
            "mean_track_1_trial_02.csv",
            "mean_track_1_trial_03.csv",
            "mean_track_2_trial_01.csv",
            "mean_track_2_trial_02.csv",
            "mean_track_2_trial_03.csv",
            "mean_track_2_trial_04.csv",
            "mean_track_3_trial_01.csv",
            "mean_track_3_trial_02.csv"
        ]
        
        for fileName, expectedFileContent \
                in zip(expectedListOfCsvFiles, expectedFileContents):
            fullPathToFile = os.path.join(self.pathToChartsDir, fileName)
            self._assertFileContent(fullPathToFile, expectedFileContent)

    def test_ExportStdevComparisonToCsv(self):
        fullFilledStdevFitness = {
            "RaceTrack_1" : {
                "DE": [ [2.14, 3.57, 14.03],
                    [2.34, 3.7, 5.13, 14.11],
                    [3.31, 15.03] ],
                "PSO_Pbest" : [ [1.95, 4.21, 5.13, 14.99],
                    [1.74, 8.13, 14.56],
                    [1.23, 5.67, 9.99, 14.0] ],
                "PSO_Episode" : [ [1.34, 3.45, 4.91, 12.33],
                    [1.5, 7.33, 13.78],
                    [0.99, 4.21, 8.88, 13.5] ]
            },
            "RaceTrack_2" : {
                "DE": [ [2.13, 7.57, 13.07],
                    [2.73, 7.7, 5.17, 13.11],
                    [7.71, 15.07],
                    [3.14, 5.68, 9.46, 14.13]],
                "PSO_Pbest" : [ [1.95, 3.21, 5.17, 13.99],
                    [1.73, 8.17, 13.56],
                    [1.27, 5.67, 9.99, 13.0],
                    [2.14, 6.8, 9.7, 15.3]],
                "PSO_Episode" : [ [1.95, 2.21, 4.17, 12.79],
                    [1.23, 5.67, 9.99],
                    [0.81, 4.21, 8.51, 12.1],
                    [1.23, 5.67, 7.7, 11.1]],
            },
            "RaceTrack_3" : {
                "DE": [ [2.14, 3.57, 4.13, 5.11, 6.18, 6.48, 6.7, 8.13, 9.27, \
                        11.35, 12.48, 13.57, 14.03, 14.35, 15.78, 17.33, 19.21],
                    [2.73, 7.7, 5.17, 13.11]],
                "PSO_Pbest" : [ [1.95, 4.21, 5.13, 14.99, 20.0, 20.5, 21.3, \
                        22.4, 23.1, 24.2, 24.6, 25.6, 26.11],
                    [7.71, 15.07] ],
                "PSO_Episode" : [ [1.35, 3.21, 4.13, 12.99, 19.0, 19.5, 20.3, \
                        21.4, 22.1, 23.2, 23.6, 24.6, 25.11],
                    [5.71, 13.07] ]
            }
        }
        self._generator._dataCollector.StdevFitness = fullFilledStdevFitness
        self._generator.ExportStdevComparisonToCsv()
        
        expectedFileContents = [
            "DE,2.14,3.57,14.03\nPSO_Pbest,1.95,4.21,5.13,14.99\n" \
                "PSO_Episode,1.34,3.45,4.91,12.33\n",
            "DE,2.34,3.7,5.13,14.11\nPSO_Pbest,1.74,8.13,14.56\n" \
                "PSO_Episode,1.5,7.33,13.78\n",
            "DE,3.31,15.03\nPSO_Pbest,1.23,5.67,9.99,14.0\n" \
                "PSO_Episode,0.99,4.21,8.88,13.5\n",
            "DE,2.13,7.57,13.07\nPSO_Pbest,1.95,3.21,5.17,13.99\n" \
                "PSO_Episode,1.95,2.21,4.17,12.79\n",
            "DE,2.73,7.7,5.17,13.11\nPSO_Pbest,1.73,8.17,13.56\n" \
                "PSO_Episode,1.23,5.67,9.99\n",
            "DE,7.71,15.07\nPSO_Pbest,1.27,5.67,9.99,13.0\n" \
                "PSO_Episode,0.81,4.21,8.51,12.1\n",
            "DE,3.14,5.68,9.46,14.13\nPSO_Pbest,2.14,6.8,9.7,15.3\n" \
                "PSO_Episode,1.23,5.67,7.7,11.1\n",
            "DE,2.14,3.57,4.13,5.11,6.18,6.48,6.7,8.13,9.27," \
                "11.35,12.48,13.57,14.03,14.35,15.78,17.33,19.21\n" \
                "PSO_Pbest,1.95,4.21,5.13,14.99,20.0,20.5,21.3,22.4,23.1,24.2," \
                "24.6,25.6,26.11\nPSO_Episode,1.35,3.21,4.13,12.99,19.0,19.5," \
                "20.3,21.4,22.1,23.2,23.6,24.6,25.11\n",
            "DE,2.73,7.7,5.17,13.11\nPSO_Pbest,7.71,15.07\n" \
                "PSO_Episode,5.71,13.07\n"
        ]
        
        expectedListOfCsvFiles = [
            "stdev_track_1_trial_01.csv",
            "stdev_track_1_trial_02.csv",
            "stdev_track_1_trial_03.csv",
            "stdev_track_2_trial_01.csv",
            "stdev_track_2_trial_02.csv",
            "stdev_track_2_trial_03.csv",
            "stdev_track_2_trial_04.csv",
            "stdev_track_3_trial_01.csv",
            "stdev_track_3_trial_02.csv"
        ]

        for fileName, expectedFileContent \
                in zip(expectedListOfCsvFiles, expectedFileContents):
            fullPathToFile = os.path.join(self.pathToChartsDir, fileName)
            self._assertFileContent(fullPathToFile, expectedFileContent)

    def test_exportChartDataToCsv(self):
        pathToNewCsv = os.path.join(self.pathToChartsDir, "TEST_DATA.csv")
        lineLabels = ["DE", "PSO"]
        chartData = [
            [1.21, 1.45678, 2.45975, 2.459875, 3,426785],
            [2.24598, 5.48426, 6.45216, 7.246821, 8.4897526]
        ]
        expectedFileContent = \
                "DE,1.21,1.45678,2.45975,2.459875,3,426785\n" \
                "PSO,2.24598,5.48426,6.45216,7.246821,8.4897526\n"
        
        self._generator._exportChartDataToCsv(
                pathToNewCsv,
                lineLabels,
                chartData)
        
        doesCsvFileExist = os.path.isfile(pathToNewCsv)
        self.assertTrue(doesCsvFileExist)
        
        with open(pathToNewCsv, "r") as actualCsvFile:
            actualFileContent = actualCsvFile.read()
            self.assertEqual(actualFileContent, expectedFileContent)

    def test_CreateMeanTrainingTimeCharts(self):
        fullFilledMeanTrainingTimesAsSeconds = {
            "RaceTrack_1" : {
                "DE": {
                    "Sum" : 600.12345567,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 540.5165789,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 800.2648972,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 765.9449842,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 999.1226489,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 987.4537895,
                    "TrialCounter" : 10
                }
            }
        }
        fullFilledMeanTrainingTimesAsEpisodes = {
            "RaceTrack_1" : {
                "DE": {
                    "Sum" : 60.51384,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 54.13,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 80.695,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 76.1,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 99.83,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 98.11,
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
                    "Sum" : 600.12345567,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 540.5165789,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_2" : {
                "DE": {
                    "Sum" : 800.2648972,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 765.9449842,
                    "TrialCounter" : 10
                }
            },
            "RaceTrack_3" : {
                "DE": {
                    "Sum" : 999.1226489,
                    "TrialCounter" : 10
                },
                "PSO" : {
                    "Sum" : 989.4537895,
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
