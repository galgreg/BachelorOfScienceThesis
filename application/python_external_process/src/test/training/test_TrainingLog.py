from src.training.TrainingLog import *
from ddt import ddt, data, unpack
from io import StringIO
import os
import os.path
import unittest
from unittest.mock import patch
from shutil import rmtree

@ddt
class TestTrainingLog(unittest.TestCase):
    @unpack
    @data((True, True, ""), (False, False, "run_log"))
    def test_Constructor(
            self,
            expectedVerboseValue,
            shouldContructWithDefaultFileName,
            logFileName):
        trainingLog = None
        if shouldContructWithDefaultFileName:
            trainingLog = TrainingLog(expectedVerboseValue)
        else:
            trainingLog = TrainingLog(expectedVerboseValue, logFileName)
        
        actualVerboseValue = trainingLog._isVerbose
        self.assertEqual(actualVerboseValue, expectedVerboseValue)
        
        expectedFileName = None
        if shouldContructWithDefaultFileName:
            expectedFileName = "training_log"
        else:
            expectedFileName = logFileName
        
        actualFileName = trainingLog._fileName
        self.assertEqual(actualFileName, expectedFileName)
        
        expectedLogContent = ""
        actualLogContent = trainingLog._content
        self.assertEqual(actualLogContent, expectedLogContent)

    @unpack
    @data((True, "Peugeot 106 Rallye 1.4 75KM", "Peugeot 106 Rallye 1.4 75KM\n"),
            (False, "Peugeot 106 Rallye 1.4 75KM", ""))
    def test_Append_IsVerbose(
            self,
            isLogVerbose,
            infoToAppend,
            expectedPrintMessage):
        trainingLog = TrainingLog(isLogVerbose)
        logContentBeforeCall = "Ford Sierra II 2.0 DOHC 125KM\n"
        trainingLog._content = logContentBeforeCall
        
        with patch('sys.stdout', new=StringIO()) as fakeOutput:
            trainingLog.Append(infoToAppend)
            actualPrintMessage = fakeOutput.getvalue()
            self.assertEqual(actualPrintMessage, expectedPrintMessage)
        
        expectedLogContentAfterCall = logContentBeforeCall + infoToAppend + "\n"
        actualLogContentAfterCall = trainingLog._content
        self.assertEqual(actualLogContentAfterCall, expectedLogContentAfterCall)

    @unpack
    @data((True, 1), (True, ""), (True, None), (True, [1, 2]), (True, {"a": "b"}),
            (False, 1), (False, ""), (False, None), (False, [1, 2]), (False, {"a": "b"}))
    def test_Append_InvalidParameters(self, isLogVerbose, invalidInfoParameter):
        trainingLog = TrainingLog(isLogVerbose)
        
        logContentBeforeCall = "Ford Sierra II 2.0 DOHC 125KM\n"
        trainingLog._content = logContentBeforeCall
        
        with patch('sys.stdout', new=StringIO()) as fakeOutput:
            trainingLog.Append(invalidInfoParameter)
            expectedPrintMessage = ""
            actualPrintMessage = fakeOutput.getvalue()
            self.assertEqual(actualPrintMessage, expectedPrintMessage)
        
        logContentAfterCall = trainingLog._content
        self.assertEqual(logContentAfterCall, logContentBeforeCall)

    def test_Save_OK(self):
        testLocation = "TEST_LOCATION_FOR_LOG_FILE"
        if os.path.isdir(testLocation):
            rmtree(testLocation)
        
        os.mkdir(testLocation)
        self.assertTrue(os.path.isdir(testLocation))
        
        trainingLog = TrainingLog(False)
        trainingLog._content = "Ford Sierra II 2.0 DOHC 125KM"
        trainingLog.Save(testLocation)
        
        pathToLogFile = os.path.join(
                testLocation,
                "{0}.txt".format(trainingLog._fileName))
        self.assertTrue(os.path.isfile(pathToLogFile))
        
        with open(pathToLogFile, "r") as logFile:
            expectedLogFileContent = trainingLog._content
            actualLogFileContent = logFile.read()
            self.assertEqual(actualLogFileContent, expectedLogFileContent)
        
        rmtree(testLocation)
        self.assertFalse(os.path.isdir(testLocation))
    
    def test_Save_LocationIsNotDir(self):
        nonDirectoryLocation = "TEST_NON_DIRECTORY_LOCATION.txt"
        with open(nonDirectoryLocation, "w") as tempFile:
            tempFile.write("kanapka")
        self.assertTrue(os.path.exists(nonDirectoryLocation))
        self.assertFalse(os.path.isdir(nonDirectoryLocation))
        
        trainingLog = TrainingLog(False)
        trainingLog._content = "Ford Sierra II 2.0 DOHC 125KM"
        trainingLog.Save(nonDirectoryLocation)
        
        pathToLogFile = os.path.os.path.join(
                nonDirectoryLocation,
                "%s.txt" % trainingLog._fileName)
        doesLogFileExist = os.path.exists(pathToLogFile)
        self.assertFalse(doesLogFileExist)
        os.remove(nonDirectoryLocation)
        self.assertFalse(os.path.exists(nonDirectoryLocation))
    
    def test_Save_LocationDoesNotExist(self):
        nonExistentLocation = "TEST_NON_EXISTENT_LOCATION"
        self.assertFalse(os.path.exists(nonExistentLocation))
        trainingLog = TrainingLog(False)
        trainingLog._content = "Ford Sierra II 2.0 DOHC 125KM"
        trainingLog.Save(nonExistentLocation)
        
        pathToLogFile = os.path.join(
                nonExistentLocation,
                "%s.txt" % trainingLog._fileName)
        doesLogFileExist = os.path.exists(pathToLogFile)
        self.assertFalse(doesLogFileExist)
    
    @data(None, 1, 1.0, [1, 2, 3, 4], {"not" : "string"})
    def test_Save_LocationIsNotString(self, notStringLocation):
        trainingLog = TrainingLog(False)
        trainingLog._content = "Ford Sierra II 2.0 DOHC 125KM"
        trainingLog.Save(notStringLocation)
        
        pathToLogFile = os.path.join(
                str(notStringLocation),
                "%s.txt" % trainingLog._fileName)
        doesLogFileExist = os.path.exists(pathToLogFile)
        self.assertFalse(doesLogFileExist)
    
    @data("", "    ", "\n \n \t")
    def test_Save_LogContentIsEmptyOrOnlyWhitespaces(self, logContent):
        testLocation = "TEST_LOCATION_FOR_LOG_FILE"
        if os.path.isdir(testLocation):
            rmtree(testLocation)
        
        os.mkdir(testLocation)
        self.assertTrue(os.path.isdir(testLocation))
        
        trainingLog = TrainingLog(False)
        trainingLog._content = logContent
        trainingLog.Save(testLocation)
        
        pathToLogFile = os.path.join(
                testLocation,
                "{0}.txt".format(trainingLog._fileName))
        self.assertFalse(os.path.exists(pathToLogFile))
        
        rmtree(testLocation)
        self.assertFalse(os.path.isdir(testLocation))
