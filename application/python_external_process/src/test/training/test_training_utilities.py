import json
from os import remove
from src.training.training_utilities import *
from ddt import ddt, data, unpack
import unittest
from unittest.mock import patch

@ddt
class TestTrainingUtilities(unittest.TestCase):
    @data(None, 1.1, 13, [1, 2, 3, 4], {"invalid" : "path"}, "", \
            "NON_EXISTENT_FILE.json")
    def test_loadConfigData_InvalidFilePaths(self, invalidFilePath):
        expectedConfigData = {}
        actualConfigData = loadConfigData(invalidFilePath)
        self.assertEqual(actualConfigData, expectedConfigData)
    
    def test_loadConfigData_FileHasWrongExtension(self):
        pathWithWrongExtension = "FILE_WITH_WRONG_EXTENSION.txt"
        with open(pathWithWrongExtension, "w") as configFile:
            configFile.write('{"kanapka" : true}')
        
        expectedConfigData = {}
        actualConfigData = loadConfigData(pathWithWrongExtension)
        self.assertEqual(actualConfigData, expectedConfigData)
        remove(pathWithWrongExtension)
    
    def test_loadConfigData_EverythingIsOK(self):
        jsonContent = \
'''
{
    "zupa_rybna" : {
        "makaron" : true,
        "ile_ryb" : 5,
        "gatunki_ryb" : ["sum", "szprot", "makrela"]
    }
}
'''
        tempTestFilePath = "TEST_FILE_PATH.json"
        with open(tempTestFilePath, "w") as configFile:
            configFile.write(jsonContent)
        
        expectedConfigData = json.loads(jsonContent)
        actualConfigData = loadConfigData(tempTestFilePath)
        self.assertEqual(actualConfigData, expectedConfigData)
        remove(tempTestFilePath)
