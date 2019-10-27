from src.AgentNeuralNetwork import *
from src.Logger import *
from datetime import datetime
from fnmatch import fnmatch
import torch
import os
import os.path
from shutil import rmtree

class TrainingResultsRepository:
    def __init__(self, trainingLog = None):
        self._trainingLog = trainingLog

    def Save(self, population, bestIndividual, shouldSavePopulation):
        if self._trainingLog is None:
            self._trainingLog = Logger(isVerbose = False)
            self._trainingLog.Append(
                    "TrainingResultsRepository.Save() warning: " \
                    "trainingLog was None! Potentially important details about " \
                    "training (or run) could haven't been saved!")
        
        locationForTrainingResults = self._createLocationForTrainingResults()
        os.mkdir(locationForTrainingResults)
        if self._doParametersHaveValidTypes(
                population,
                bestIndividual,
                shouldSavePopulation):
            if len(population) == 0:
                self._trainingLog.Append(
                        "TrainingResultsRepository.Save() error: " \
                        "population is empty!")
            else:
                self._saveBestModel(
                        locationForTrainingResults,
                        bestIndividual)
                if shouldSavePopulation:
                    self._saveWholePopulation(
                            locationForTrainingResults,
                            population)
                self._trainingLog.Append(
                        "TrainingResultsRepository.Save() info: " \
                        "training results were saved to the location " \
                        "'{0}'!".format(locationForTrainingResults))
        else:
            self._trainingLog.Append(
                    "TrainingResultsRepository.Save() error: " \
                    "some of parameters have wrong type!\n" \
                    "type(population) == {0}, " \
                    "type(bestIndividual) == {1}, " \
                    "type(shouldSavePopulation) == {2}".format(
                            type(population),
                            type(bestIndividual),
                            type(shouldSavePopulation)))
        
        self._trainingLog.Save(locationForTrainingResults)
    
    def LoadBestModel(self, dirNameWithModelToLoad):
        if self._trainingLog is None:
            self._trainingLog = Logger(isVerbose = False)
            self._trainingLog.Append(
                    "TrainingResultsRepository.LoadBestModel() warning: " \
                    "trainingLog was None! Potentially important details about" \
                    " training (or run) could haven't been saved!")
        
        bestModel = None
        if type(dirNameWithModelToLoad) == str:
            basePathToModel = self._createBasePathForResults()
            fullPathToModel = \
                    os.path.join(
                            basePathToModel,
                            dirNameWithModelToLoad,
                            "best_model.pth")
            if os.path.isfile(fullPathToModel):
                bestModel = torch.load(fullPathToModel)
                self._trainingLog.Append(
                        "TrainingResultsRepository.LoadBestModel() info: " \
                        "'training_results/{0}/best_model.pth' file has been "
                        "loaded!".format(dirNameWithModelToLoad))
            else:
                self._trainingLog.Append(
                        "TrainingResultsRepository.LoadBestModel() error: " \
                        "cannot load 'best_model.pth' file - path does not" \
                        " exist! (dirname = '{0}')".format(dirNameWithModelToLoad))
        else:
            self._trainingLog.Append(
                    "TrainingResultsRepository.LoadBestModel() error: " \
                    "dirNameWithModelToLoad has wrong type! " \
                    "(expected: str, actual: {0})".format(
                            type(dirNameWithModelToLoad)))
        return bestModel
    
    def LoadPopulation(self, dirNameWithPopulationToLoad):
        if self._trainingLog is None:
            self._trainingLog = Logger(isVerbose = False)
            self._trainingLog.Append(
                    "TrainingResultsRepository.LoadPopulation() warning: " \
                    "trainingLog was None! Potentially important details about" \
                    " training (or run) could haven't been saved!")
        
        population = None
        if type(dirNameWithPopulationToLoad) == str:
            basePathToPopulation = self._createBasePathForResults()
            fullPathToPopulation = \
                    os.path.join(
                            basePathToPopulation,
                            dirNameWithPopulationToLoad,
                            "population")
            if os.path.isdir(fullPathToPopulation):
                models = []
                listOfModelFileNames = os.listdir(fullPathToPopulation)
                listOfModelFileNames.sort()
                for fileName in listOfModelFileNames:
                    if fnmatch(fileName, "model_*.pth"):
                        fullPathToModelFile = \
                                os.path.join(fullPathToPopulation, fileName)
                                
                        tempModel = torch.load(fullPathToModelFile)
                        models.append(tempModel)
                
                if len(models) == 0:
                    self._trainingLog.Append(
                            "TrainingResultsRepository.LoadPopulation() error: " \
                            "'training_results/{0}/population' is empty - " \
                            "has no 'model_<n>.pth' files! " \
                            "(examples: 'model_1.pth', 'model_2.pth' " \
                            "etc.)".format(dirNameWithPopulationToLoad))
                else:
                    population = models
                    self._trainingLog.Append(
                            "TrainingResultsRepository.LoadPopulation() info: " \
                            "'training_results/{0}/population' has been " \
                            "loaded!".format(dirNameWithPopulationToLoad))
            else:
                self._trainingLog.Append(
                        "TrainingResultsRepository.LoadPopulation() error: " \
                        "cannot load population from " \
                        "'training_results/{0}/population' - path " \
                        "does not exist!".format(dirNameWithPopulationToLoad))
        else:
            self._trainingLog.Append(
                    "TrainingResultsRepository.LoadPopulation() error: " \
                    "dirNameWithPopulationToLoad has wrong type! " \
                    "(expected: str, actual: {0})".format(
                            type(dirNameWithPopulationToLoad)))
        
        return population
    
    def _doParametersHaveValidTypes(
            self,
            population,
            bestIndividual,
            shouldSavePopulation):
        return type(population) == list \
                and type(bestIndividual) == AgentNeuralNetwork \
                and type(shouldSavePopulation) == bool
    
    def _createLocationForTrainingResults(self):
        basePath = self._createBasePathForResults()
        if not os.path.isdir(basePath):
            os.mkdir(basePath)
        dirName = self._createDirNameForResults()
        fullPathToLocation = os.path.join(basePath, dirName)
        return fullPathToLocation
    
    def _createBasePathForResults(self):
        basePath = os.path.dirname(os.path.realpath(__file__))
        basePathLastPart = "python_external_process"
        basePathEnd = \
                basePath.index(basePathLastPart) + len(basePathLastPart)        
        basePath = basePath[ : basePathEnd]
        basePath = os.path.join(basePath, "training_results")
        return basePath
    
    def _createDirNameForResults(self):
        currentDateTime = datetime.now()
        dirName = "{0}_{1}_{2}_{3}_{4}_{5}".format(
                str(currentDateTime.year).zfill(2),
                str(currentDateTime.month).zfill(2),
                str(currentDateTime.day).zfill(2),
                str(currentDateTime.hour).zfill(2),
                str(currentDateTime.minute).zfill(2),
                str(currentDateTime.second).zfill(2)
        )
        return dirName

    def _saveBestModel(self, location, bestModel):
        bestModelFileName = "best_model.pth"
        fullPathForModelFile = os.path.join(location, bestModelFileName)
        torch.save(bestModel, fullPathForModelFile)

    def _saveWholePopulation(self, location, population):
        dirForPopulation = "population"
        locationForPopulationFiles = os.path.join(location, dirForPopulation)
        
        if os.path.isdir(locationForPopulationFiles):
            rmtree(locationForPopulationFiles)
        os.mkdir(locationForPopulationFiles)
        
        for i in range(len(population)):
            fileNameForModel = "model_{0}.pth".format(str(i+1).zfill(3))
            fullPathForModel = \
                    os.path.join(locationForPopulationFiles, fileNameForModel)
            torch.save(population[i], fullPathForModel)
