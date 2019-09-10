from src.training.TrainingResultsRepository import *
from src.training.TrainingLog import *
from src.AgentsPopulation import *
from src.AgentNeuralNetwork import *
from ddt import ddt, data, unpack
import os
import os.path
from shutil import rmtree
import torch
import unittest
from unittest.mock import patch

@ddt
class TestTrainingResultsRepository(unittest.TestCase):
    def setUp(self):
        self._trainingLog = TrainingLog(isVerbose = False)
        self._repository = TrainingResultsRepository(self._trainingLog)
    
    def tearDown(self):
        del self._repository
        del self._trainingLog
    
    @patch('src.training.TrainingResultsRepository.datetime')
    def test_Save_OK_SaveWithoutPopulation(self, mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        locationForResults = \
                self._createExpectedLocationForResults(mock_datetime)
        if os.path.isdir(locationForResults):
            rmtree(locationForResults)
        
        population = AgentsPopulation(2, [2, 2], None)
        whichModelIsTheBest = 1
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            self._repository.Save(population, whichModelIsTheBest, False)
        
        doesLocationExist = os.path.isdir(locationForResults)
        self.assertTrue(doesLocationExist)
        
        pathToLogFile = os.path.join(locationForResults, "training.log")
        doesLogFileExist = os.path.isfile(pathToLogFile)
        self.assertTrue(doesLogFileExist)
        
        expectedLogContent = \
                "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.Save() info: " \
                "training results were saved to the location '{0}'!\n".format(
                        locationForResults)
        
        with open(pathToLogFile, "r") as logFile:
            actualLogContent = logFile.read()
            self.assertEqual(actualLogContent, expectedLogContent)
        
        pathToBestModel = os.path.join(locationForResults, "best_model.pth")
        doesBestModelExist = os.path.isfile(pathToBestModel)
        self.assertTrue(doesBestModelExist)
        
        expectedBestModel = population._agents[whichModelIsTheBest]
        actualBestModel = torch.load(pathToBestModel)
        self._compareModels(actualBestModel, expectedBestModel)
        
        pathToPopulation = os.path.join(locationForResults, "population")
        doesPopulationExist = os.path.isdir(pathToPopulation)
        self.assertFalse(doesPopulationExist)

        rmtree(locationForResults)
    
    @patch('src.training.TrainingResultsRepository.datetime')
    def test_Save_OK_SaveWithPopulation(self, mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        locationForResults = \
                self._createExpectedLocationForResults(mock_datetime)
        if os.path.isdir(locationForResults):
            rmtree(locationForResults)
        
        population = AgentsPopulation(2, [2, 2], None)
        whichModelIsTheBest = 1
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            self._repository.Save(population, whichModelIsTheBest, True)
        
        doesLocationExist = os.path.isdir(locationForResults)
        self.assertTrue(doesLocationExist)
        
        pathToLogFile = os.path.join(locationForResults, "training.log")
        doesLogFileExist = os.path.isfile(pathToLogFile)
        self.assertTrue(doesLogFileExist)
        
        expectedLogContent = \
                "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.Save() info: " \
                "training results were saved to the location '{0}'!\n".format(
                        locationForResults)
        
        with open(pathToLogFile, "r") as logFile:
            actualLogContent = logFile.read()
            self.assertEqual(actualLogContent, expectedLogContent)
        
        pathToBestModel = os.path.join(locationForResults, "best_model.pth")
        doesBestModelExist = os.path.isfile(pathToBestModel)
        self.assertTrue(doesBestModelExist)
        
        expectedBestModel = population._agents[whichModelIsTheBest]
        actualBestModel = torch.load(pathToBestModel)
        self._compareModels(actualBestModel, expectedBestModel)
        
        pathToPopulation = os.path.join(locationForResults, "population")
        doesPopulationExist = os.path.isdir(pathToPopulation)
        self.assertTrue(doesPopulationExist)
        
        for i in range(len(population._agents)):
            pathToModelFile = \
                    os.path.join(
                            pathToPopulation,
                            "model_{0}.pth".format(str(i+1).zfill(3)))
            doesModelFileExist = os.path.exists(pathToModelFile)
            self.assertTrue(doesModelFileExist)
            
            expectedModel = population._agents[i]
            actualModel = torch.load(pathToModelFile)
            self._compareModels(actualModel, expectedModel)
        
        rmtree(locationForResults)
    
    @unpack
    @data((None, None, None), (None, None, True), (None, 5, None), \
            (AgentsPopulation(2, [2, 2], None), None, None), \
            (AgentsPopulation(2, [2, 2], None), 1, None), \
            (AgentsPopulation(2, [2, 2], None), None, True),
            ([1, 2, 3, 4], 2, True), \
            (AgentsPopulation(2, [2, 2], None), 1.2, True), \
            (AgentsPopulation(2, [2, 2], None), 1, "kanapka"))
    @patch('src.training.TrainingResultsRepository.datetime')
    def test_Save_SomeParametersHaveWrongType(
            self,
            population,
            whichModelIsTheBest,
            shouldSavePopulation,
            mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        expectedLogContent = \
                "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.Save() error: " \
                "some of parameters have wrong type!\n" \
                "type(population) == {0}, " \
                "type(whichModelIsTheBest) == {1}, " \
                "type(shouldSavePopulation) == {2}\n".format(
                        type(population),
                        type(whichModelIsTheBest),
                        type(shouldSavePopulation)
                )
        self._assertSaveTestForInvalidParameters(
                population,
                whichModelIsTheBest,
                shouldSavePopulation,
                mock_datetime,
                expectedLogContent)
    
    @patch('src.training.TrainingResultsRepository.datetime')
    def test_Save_EmptyPopulation(self, mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        expectedLogContent = \
                "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.Save() error: population is empty!\n"
        self._assertSaveTestForInvalidParameters(
                population = AgentsPopulation(0, [2, 2], None),
                whichModelIsTheBest = 0,
                shouldSavePopulation = True,
                mock_datetime = mock_datetime,
                expectedLogContent = expectedLogContent)
    
    @data(-1, 11, 100)
    @patch('src.training.TrainingResultsRepository.datetime')
    def test_Save_BestModelIsOutOfAllowedRange(
            self,
            whichModelIsBest,
            mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        expectedLogContent = \
                "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.Save() error: " \
                "whichModelIsBest is out of allowed range!\n" \
                "Allowed range is from 0 to 9 (inclusive)!\n"
        self._assertSaveTestForInvalidParameters(
                population = AgentsPopulation(10, [2, 2], None),
                whichModelIsTheBest = whichModelIsBest,
                shouldSavePopulation = True,
                mock_datetime = mock_datetime,
                expectedLogContent = expectedLogContent)

    @patch('src.training.TrainingResultsRepository.datetime')
    def test_Save_TrainingLogWasNone(self, mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        locationForResults = \
                self._createExpectedLocationForResults(mock_datetime)
        if os.path.isdir(locationForResults):
            rmtree(locationForResults)
        
        population = AgentsPopulation(2, [2, 2], None)
        whichModelIsTheBest = 1
        
        self._repository = TrainingResultsRepository(trainingLog = None)
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            self._repository.Save(population, whichModelIsTheBest, False)
        
        doesLocationExist = os.path.isdir(locationForResults)
        self.assertTrue(doesLocationExist)
        
        pathToLogFile = os.path.join(locationForResults, "training.log")
        doesLogFileExist = os.path.isfile(pathToLogFile)
        self.assertTrue(doesLogFileExist)
        
        expectedLogContent = \
                "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.Save() warning: " \
                "trainingLog was None! Potentially important details about " \
                "training (or run) could haven't been saved!\n" \
                "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.Save() info: " \
                "training results were saved to the location " \
                "'{0}'!\n".format(locationForResults)
        
        with open(pathToLogFile, "r") as logFile:
            actualLogContent = logFile.read()
            self.assertEqual(actualLogContent, expectedLogContent)
        
        pathToBestModel = os.path.join(locationForResults, "best_model.pth")
        doesBestModelExist = os.path.isfile(pathToBestModel)
        self.assertTrue(doesBestModelExist)
        
        expectedBestModel = population._agents[whichModelIsTheBest]
        actualBestModel = torch.load(pathToBestModel)
        self._compareModels(actualBestModel, expectedBestModel)
        
        pathToPopulation = os.path.join(locationForResults, "population")
        doesPopulationExist = os.path.isdir(pathToPopulation)
        self.assertFalse(doesPopulationExist)

        rmtree(locationForResults)
    
    def _assertSaveTestForInvalidParameters(
            self,
            population,
            whichModelIsTheBest,
            shouldSavePopulation,
            mock_datetime,
            expectedLogContent):
        locationForResults = \
                self._createExpectedLocationForResults(mock_datetime)
        if os.path.isdir(locationForResults):
            rmtree(locationForResults)
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            self._repository.Save(
                    population,
                    whichModelIsTheBest,
                    shouldSavePopulation)
        
        doesLocationExist = os.path.isdir(locationForResults)
        self.assertTrue(doesLocationExist)
        
        pathToLogFile = os.path.join(locationForResults, "training.log")
        doesLogFileExist = os.path.isfile(pathToLogFile)
        self.assertTrue(doesLogFileExist)

        with open(pathToLogFile, "r") as logFile:
            actualLogContent = logFile.read()
            self.assertEqual(actualLogContent, expectedLogContent)
        
        pathToBestModel = os.path.join(locationForResults, "best_model.pth")
        doesBestModelExist = os.path.isfile(pathToBestModel)
        self.assertFalse(doesBestModelExist)
        
        pathToPopulation = os.path.join(locationForResults, "population")
        doesPopulationExist = os.path.isdir(pathToPopulation)
        self.assertFalse(doesPopulationExist)
        
        rmtree(locationForResults)
    
    @unpack
    @data((TrainingLog(isVerbose = False), ""), \
        (None, "[ 1995-07-04 17:15:00 ] TrainingResultsRepository.LoadBestModel() " \
                "warning: trainingLog was None! Potentially important details" \
                " about training (or run) could haven't been saved!\n"))
    @patch('src.training.TrainingResultsRepository.datetime')
    def test_LoadBestModel_OK(self, trainingLog, logWarning, mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        locationForModel = self._createExpectedLocationForResults(mock_datetime)
        if os.path.isdir(locationForModel):
            rmtree(locationForModel)
        
        os.mkdir(locationForModel)
        doesLocationForModelExist = os.path.isdir(locationForModel)
        self.assertTrue(doesLocationForModelExist)
        
        pathToBestModelFile = os.path.join(locationForModel, "best_model.pth")
        tempBestModel = AgentNeuralNetwork([5, 3, 2])
        torch.save(tempBestModel, pathToBestModelFile)
        doesBestModelExist = os.path.isfile(pathToBestModelFile)
        self.assertTrue(doesBestModelExist)

        dirNameForModel = "1995_07_04_17_15_00"
        expectedModel = tempBestModel
        self._repository._trainingLog = trainingLog
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            actualModel = self._repository.LoadBestModel(dirNameForModel)
        
        expectedLogContent = \
                logWarning + "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadBestModel() info:" \
                " training_results/{0}/best_model.pth file has been loaded!" \
                "\n".format(dirNameForModel)
        actualLogContent = self._repository._trainingLog._content
        self.assertEqual(actualLogContent, expectedLogContent)
        
        self._compareModels(actualModel, expectedModel)
        rmtree(locationForModel)
    
    @unpack
    @data((None, TrainingLog(isVerbose = False), ""), \
        ([1, 2, 3, 4], TrainingLog(isVerbose = False), ""),
        (1.1, TrainingLog(isVerbose = False), ""),
        ({"wrong" : "type"}, TrainingLog(isVerbose = False), ""),
        (None, None, "[ 1995-07-04 17:15:00 ] TrainingResultsRepository" \
                ".LoadBestModel() warning: trainingLog was None! Potentially " \
                "important details about training (or run) could haven't been " \
                "saved!\n"),
        ([1, 2, 3, 4], None, "[ 1995-07-04 17:15:00 ] TrainingResultsRepository" \
                ".LoadBestModel() warning: trainingLog was None! Potentially " \
                "important details about training (or run) could haven't been " \
                "saved!\n"),
        (1.1, None, "[ 1995-07-04 17:15:00 ] TrainingResultsRepository" \
                ".LoadBestModel() warning: trainingLog was None! Potentially " \
                "important details about training (or run) could haven't been " \
                "saved!\n"),
        ({"wrong" : "type"}, None, "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadBestModel() warning: " \
                "trainingLog was None! Potentially important details about " \
                "training (or run) could haven't been saved!\n"))
    def test_LoadBestModel_DirNameHasWrongType(
            self,
            dirName,
            trainingLog,
            logWarning):
        self._repository._trainingLog = trainingLog
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            bestModel = self._repository.LoadBestModel(dirName)
        
        self.assertTrue(bestModel is None)
        expectedLog = \
                logWarning + "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadBestModel() error: " \
                "dirNameWithModelToLoad has wrong type! " \
                "(expected: str, actual: {0})\n".format(type(dirName))
        actualLog = self._repository._trainingLog._content
        self.assertEqual(actualLog, expectedLog)
    
    @unpack
    @data((TrainingLog(isVerbose = False), ""), \
        (None, "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadBestModel() warning: " \
                "trainingLog was None! Potentially important details about " \
                "training (or run) could haven't been saved!\n"))
    def test_LoadBestModel_DirNameDoesNotExist(self, trainingLog, logWarning):
        nonExistentDirName = "TEST_NON_EXISTENT_DIR_NAME"
        self._repository._trainingLog = trainingLog
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            bestModel = self._repository.LoadBestModel(nonExistentDirName)
        
        self.assertTrue(bestModel is None)
        expectedLogContent = \
                logWarning + "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadBestModel() error: " \
                "cannot load 'best_model.pth' file - path does not exist! " \
                "(dirname = '{0}')\n".format(nonExistentDirName)
        actualLogContent = self._repository._trainingLog._content
        self.assertEqual(actualLogContent, expectedLogContent)
    
    @unpack
    @data((TrainingLog(isVerbose = False), ""), \
        (None, "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadBestModel() warning: " \
                "trainingLog was None! Potentially important details about " \
                "training (or run) could haven't been saved!\n"))
    @patch('src.training.TrainingResultsRepository.datetime')
    def test_LoadBestModel_FileForBestModelDoesNotExist(
            self,
            trainingLog,
            logWarning,
            mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        locationForModel = self._createExpectedLocationForResults(mock_datetime)
        if os.path.isdir(locationForModel):
            rmtree(locationForModel)
        
        os.mkdir(locationForModel)
        doesLocationForModelExist = os.path.isdir(locationForModel)
        self.assertTrue(doesLocationForModelExist)
        
        pathToBestModelFile = os.path.join(locationForModel, "best_model.pth")
        doesBestModelExist = os.path.isfile(pathToBestModelFile)
        self.assertFalse(doesBestModelExist)

        dirNameForModel = "1995_07_04_17_15_00"
        self._repository._trainingLog = trainingLog
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            bestModel = self._repository.LoadBestModel(dirNameForModel)
        
        self.assertTrue(bestModel is None)
        
        expectedLogContent = \
                logWarning + "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadBestModel() error: " \
                "cannot load 'best_model.pth' file - path does not exist! " \
                "(dirname = '{0}')\n".format(dirNameForModel)
        actualLogContent = self._repository._trainingLog._content
        self.assertEqual(actualLogContent, expectedLogContent)
        
        rmtree(locationForModel)
    
    @unpack
    @data((TrainingLog(isVerbose = False), ""), \
        (None, "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() warning: " \
                "trainingLog was None! Potentially important details about " \
                "training (or run) could haven't been saved!\n"))
    @patch('src.training.TrainingResultsRepository.datetime')
    def test_LoadPopulation_OK(
            self,
            trainingLog,
            logWarning,
            mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        locationForPopulation = \
                self._createExpectedLocationForResults(mock_datetime)
        if os.path.isdir(locationForPopulation):
            rmtree(locationForPopulation)
        
        os.mkdir(locationForPopulation)
        doesLocationForPopulationExist = os.path.isdir(locationForPopulation)
        self.assertTrue(doesLocationForPopulationExist)
        
        pathToPopulationFiles = os.path.join(locationForPopulation, "population")        
        os.mkdir(pathToPopulationFiles)
        doesPathToPopulationFilesExist = os.path.isdir(pathToPopulationFiles)
        self.assertTrue(doesPathToPopulationFilesExist)
        
        numberOfAgents = 5
        agentDimensions = [5, 3, 2]
        expectedPopulation = \
                AgentsPopulation(numberOfAgents, agentDimensions, None)
        
        for i in range(numberOfAgents):
            modelFileName = "model_{0}.pth".format(str(i+1).zfill(3))
            fullPathToModelFile = \
                    os.path.join(pathToPopulationFiles, modelFileName)
            torch.save(expectedPopulation._agents[i], fullPathToModelFile)
            doesModelExist = os.path.isfile(fullPathToModelFile)
            self.assertTrue(doesModelExist)
        
        self._repository._trainingLog = trainingLog
        dirNameForPopulation = "1995_07_04_17_15_00"
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            actualPopulation = \
                    self._repository.LoadPopulation(dirNameForPopulation)
        
        expectedLogContent = \
                logWarning + "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() info: " \
                "training_results/{0}/population has been loaded!\n".format(
                        dirNameForPopulation)
        actualLogContent = self._repository._trainingLog._content
        self.assertEqual(actualLogContent, expectedLogContent)
        
        rmtree(locationForPopulation)
    
    @unpack
    @data((None, TrainingLog(isVerbose = False), ""), \
        ([1, 2, 3, 4], TrainingLog(isVerbose = False), ""),
        (1.1, TrainingLog(isVerbose = False), ""),
        ({"wrong" : "type"}, TrainingLog(isVerbose = False), ""),
        (None, None, "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() warning: " \
                "trainingLog was None! Potentially important details about " \
                "training (or run) could haven't been saved!\n"),
        ([1, 2, 3, 4], None, "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() warning: " \
                "trainingLog was None! Potentially important details about " \
                "training (or run) could haven't been saved!\n"),
        (1.1, None, "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() warning: " \
                "trainingLog was None! Potentially important details about " \
                "training (or run) could haven't been saved!\n"),
        ({"wrong" : "type"}, None, "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() " \
                "warning: trainingLog was None! Potentially important details " \
                "about training (or run) could haven't been saved!\n"))
    def test_LoadPopulation_DirNameHasWrongType(
            self,
            dirName,
            trainingLog,
            logWarning):
        self._repository._trainingLog = trainingLog
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            population = self._repository.LoadPopulation(dirName)
        
        self.assertTrue(population is None)
        
        expectedLog = \
                logWarning + "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() error: " \
                "dirNameWithPopulationToLoad has wrong type! " \
                "(expected: str, actual: {0})\n".format(type(dirName))
        actualLog = self._repository._trainingLog._content
        self.assertEqual(actualLog, expectedLog)
    
    
    @unpack
    @data((TrainingLog(isVerbose = False), ""), \
        (None, "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() warning: " \
                "trainingLog was None! Potentially important details about " \
                "training (or run) could haven't been saved!\n"))
    def test_LoadPopulation_DirNameDoesNotExist(self, trainingLog, logWarning):
        nonExistentDirName = "TEST_NON_EXISTENT_DIR_NAME"
        self._repository._trainingLog = trainingLog
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            population = self._repository.LoadPopulation(nonExistentDirName)
        
        self.assertTrue(population is None)
        expectedLogContent = \
                logWarning + "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() error: " \
                "cannot load population from training_results/{0}/population -" \
                " path does not exist!\n".format(nonExistentDirName)
        actualLogContent = self._repository._trainingLog._content
        self.assertEqual(actualLogContent, expectedLogContent)
    
    @unpack
    @data((TrainingLog(isVerbose = False), ""), \
        (None, "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() warning: " \
                "trainingLog was None! Potentially important details about " \
                "training (or run) could haven't been saved!\n"))
    @patch('src.training.TrainingResultsRepository.datetime')
    def test_LoadPopulation_PopulationDirDoesNotExist(
            self,
            trainingLog,
            logWarning,
            mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        locationForPopulation = \
                self._createExpectedLocationForResults(mock_datetime)
        if os.path.isdir(locationForPopulation):
            rmtree(locationForPopulation)
        
        os.mkdir(locationForPopulation)
        doesLocationForPopulationExist = os.path.isdir(locationForPopulation)
        self.assertTrue(doesLocationForPopulationExist)
        
        pathToPopulationFiles = \
                os.path.join(locationForPopulation, "population")
        doesPathToPopulationFilesExist = os.path.isdir(pathToPopulationFiles)
        self.assertFalse(doesPathToPopulationFilesExist)
        
        self._repository._trainingLog = trainingLog
        dirName = "1995_07_04_17_15_00"
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            population = self._repository.LoadPopulation(dirName)
        
        self.assertTrue(population is None)
        
        expectedLogContent = \
                logWarning + "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() " \
                "error: cannot load population from " \
                "training_results/{0}/population - path does not " \
                "exist!\n".format(dirName)
        actualLogContent = self._repository._trainingLog._content
        self.assertEqual(actualLogContent, expectedLogContent)
    
    @unpack
    @data((TrainingLog(isVerbose = False), ""), \
        (None, "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() warning: " \
                "trainingLog was None! Potentially important details about " \
                "training (or run) could haven't been saved!\n"))
    @patch('src.training.TrainingResultsRepository.datetime')
    def test_LoadPopulation_PopulationDirIsEmpty(
            self,
            trainingLog,
            logWarning,
            mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        locationForPopulation = \
                self._createExpectedLocationForResults(mock_datetime)
        if os.path.isdir(locationForPopulation):
            rmtree(locationForPopulation)
        
        os.mkdir(locationForPopulation)
        doesLocationForPopulationExist = os.path.isdir(locationForPopulation)
        self.assertTrue(doesLocationForPopulationExist)
        
        pathToPopulationFiles = \
                os.path.join(locationForPopulation, "population")
        os.mkdir(pathToPopulationFiles)
        doesPathToPopulationFilesExist = os.path.isdir(pathToPopulationFiles)
        self.assertTrue(doesPathToPopulationFilesExist)
        
        self._repository._trainingLog = trainingLog
        dirName = "1995_07_04_17_15_00"
        
        with patch('src.training.TrainingLog.datetime') as datetime_log:
            datetime_log.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
            population = self._repository.LoadPopulation(dirName)
        
        self.assertTrue(population is None)
        
        expectedLogContent = \
                logWarning + "[ 1995-07-04 17:15:00 ] " \
                "TrainingResultsRepository.LoadPopulation() " \
                "error: training_results/{0}/population is empty - " \
                "has no 'model_<n>.pth' files! " \
                "(examples: 'model_1.pth', 'model_2.pth' etc.)\n".format(dirName)
        actualLogContent = self._repository._trainingLog._content
        self.assertEqual(actualLogContent, expectedLogContent)
    
    @unpack
    @data((None, None, None), (None, None, True), (None, 5, None), \
            (AgentsPopulation(2, [2, 2], None), None, None), \
            (AgentsPopulation(2, [2, 2], None), 1, None), \
            (AgentsPopulation(2, [2, 2], None), None, True),
            ([1, 2, 3, 4], 2, True), \
            (AgentsPopulation(2, [2, 2], None), 1.2, True), \
            (AgentsPopulation(2, [2, 2], None), 1, "kanapka"))
    def test_doParametersHaveValidTypes_False(
            self,
            population,
            whichModelIsTheBest,
            shouldSavePopulation):
        expectedResult = False
        actualResult = \
                self._repository._doParametersHaveValidTypes(
                        population,
                        whichModelIsTheBest,
                        shouldSavePopulation)
        self.assertEqual(actualResult, expectedResult)

    def test_doParametersHaveValidTypes_True(self):
        expectedResult = True
        actualResult = \
                self._repository._doParametersHaveValidTypes(
                        population = AgentsPopulation(2, [2, 2], None),
                        whichModelIsTheBest = 1,
                        shouldSavePopulation = True)
        self.assertEqual(actualResult, expectedResult)

    @patch('src.training.TrainingResultsRepository.datetime')
    def test_createLocationForTrainingResults(self, mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        expectedLocation = self._createExpectedLocationForResults(mock_datetime)
        actualLocation = self._repository._createLocationForTrainingResults()
        self.assertEqual(actualLocation, expectedLocation)
    
    def _createExpectedLocationForResults(self, mock_datetime):
        basePath = os.path.dirname(os.path.realpath(__file__))
        basePathLastPart = "python_external_process"
        basePathEnd = \
                basePath.index(basePathLastPart) + len(basePathLastPart)        
        basePath = basePath[ : basePathEnd]
        self.assertTrue(basePath.endswith(basePathLastPart))
        basePath = os.path.join(basePath, "training_results")
        
        datetimeNow = mock_datetime.now()
        dirName = "{0}_{1}_{2}_{3}_{4}_{5}".format(
                str(datetimeNow.year).zfill(2),
                str(datetimeNow.month).zfill(2),
                str(datetimeNow.day).zfill(2),
                str(datetimeNow.hour).zfill(2),
                str(datetimeNow.minute).zfill(2),
                str(datetimeNow.second).zfill(2)
        )
        return os.path.join(basePath, dirName)
    
    def test_createBasePathForResults(self):
        expectedBasePath = os.path.dirname(os.path.realpath(__file__))
        basePathLastPart = "python_external_process"
        expectedBasePathEnd = \
                expectedBasePath.index(basePathLastPart) + len(basePathLastPart)        
        expectedBasePath = expectedBasePath[ : expectedBasePathEnd]
        self.assertTrue(expectedBasePath.endswith(basePathLastPart))
        expectedBasePath = os.path.join(expectedBasePath, "training_results")
        actualBasePath = self._repository._createBasePathForResults()
        self.assertEqual(actualBasePath, expectedBasePath)

    @patch('src.training.TrainingResultsRepository.datetime')
    def test_createDirNameForResults(self, mock_datetime):
        mock_datetime.now.return_value = datetime(1995, 7, 4, 17, 15, 0)
        expectedDirName = "1995_07_04_17_15_00"
        actualDirName = self._repository._createDirNameForResults()
        self.assertEqual(actualDirName, expectedDirName)

    def test_saveBestModel_OK(self):
        population = \
                AgentsPopulation(
                        numberOfAgents = 5,
                        agentDimensions = [5, 3, 2],
                        learningAlgorithm = None)
        whichModelIsBest = 3
        
        locationForModel = "TEST_LOCATION_TO_SAVE_BEST_MODEL"
        if os.path.exists(locationForModel):
            rmtree(locationForModel)
        
        os.mkdir(locationForModel)
        self.assertTrue(os.path.isdir(locationForModel))
        
        self._repository._saveBestModel(
                locationForModel,
                population,
                whichModelIsBest)
        
        modelFilePath = os.path.join(locationForModel, "best_model.pth")
        doesModelFileExist = os.path.isfile(modelFilePath)
        self.assertTrue(doesModelFileExist)
        
        expectedSavedModel = population._agents[whichModelIsBest]
        actualSavedModel = torch.load(modelFilePath)
        self._compareModels(expectedSavedModel, actualSavedModel)
        
        rmtree(locationForModel)

    def test_saveWholePopulation_OK(self):
        population = \
                AgentsPopulation(
                        numberOfAgents = 5,
                        agentDimensions = [5, 3, 2],
                        learningAlgorithm = None)
        
        locationForPopulation = "TEST_LOCATION_TO_SAVE_BEST_POPULATION"
        if os.path.exists(locationForPopulation):
            rmtree(locationForPopulation)
        
        os.mkdir(locationForPopulation)
        self.assertTrue(os.path.isdir(locationForPopulation))
        
        self._repository._saveWholePopulation(locationForPopulation, population)
        pathToPopulationDir = os.path.join(locationForPopulation, "population")
        self.assertTrue(os.path.isdir(pathToPopulationDir))
        
        for i in range(len(population._agents)):
            fileNameForModel = "model_{0}.pth".format(str(i+1).zfill(3))
            fullPathToModelFile = \
                    os.path.join(pathToPopulationDir, fileNameForModel)
            self.assertTrue(os.path.isfile(fullPathToModelFile))
            
            expectedModel = population._agents[i]
            actualModel = torch.load(fullPathToModelFile)
            self._compareModels(actualModel, expectedModel)
        
        rmtree(locationForPopulation)

    def _compareModels(self, firstModel, secondModel):
        expectedTypeOfModel = AgentNeuralNetwork
        actualTypeOfModel_First = type(firstModel)
        self.assertEqual(actualTypeOfModel_First, expectedTypeOfModel)
        
        actualTypeOfModel_Second = type(secondModel)
        self.assertEqual(actualTypeOfModel_Second, expectedTypeOfModel)
        
        for firstLayer, secondLayer \
                in zip(firstModel._layers, secondModel._layers):
            expectedLayerType = torch.nn.Linear
            actualLayerType_First = type(firstLayer)
            actualLayerType_Second = type(secondLayer)
            
            self.assertEqual(firstLayer.in_features, secondLayer.in_features)
            self.assertEqual(firstLayer.out_features, secondLayer.out_features)
            
            firstLayerWeightParameters = firstLayer.weight.data
            secondLayerWeightParameters = secondLayer.weight.data
            self.assertTrue(
                    torch.equal(
                            firstLayerWeightParameters,
                            secondLayerWeightParameters))
            
            firstLayerBiasParameters = firstLayer.bias.data
            secondLayerBiasParameters = secondLayer.bias.data
            self.assertTrue(
                    torch.equal(
                            firstLayerBiasParameters,
                            secondLayerBiasParameters))
