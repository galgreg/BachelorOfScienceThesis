from src.AgentsPopulation import *
from src.training.TrainingLog import *
from datetime import datetime
import torch
import os
import os.path
from shutil import rmtree

# TODO -> implementacja metod LoadBestModel, LoadPopulation
# TODO -> przejrzeć gdzie brakuje zapisywania do loga
class TrainingResultsRepository:
    def __init__(self, trainingLog = None):
        self._trainingLog = trainingLog

    def Save(self, population, whichModelIsTheBest, shouldSavePopulation):
        if self._trainingLog is None:
            self._trainingLog = TrainingLog(isVerbose = False)
            self._trainingLog.Append(
                    "TrainingResultsRepository.Save() warning: " \
                    "trainingLog was None! Potentially important details about " \
                    "training (or run) could haven't been saved!")
        
        locationForTrainingResults = self._createLocationForTrainingResults()
        os.mkdir(locationForTrainingResults)
        if self._doParametersHaveValidTypes(
                population,
                whichModelIsTheBest,
                shouldSavePopulation):
            if len(population._agents) == 0:
                self._trainingLog.Append(
                        "TrainingResultsRepository.Save() error: " \
                        "population is empty!")
            elif whichModelIsTheBest < 0 \
                    or whichModelIsTheBest >= len(population._agents):
                self._trainingLog.Append(
                        "TrainingResultsRepository.Save() error: " \
                        "whichModelIsBest is out of allowed range!\n" \
                        "Allowed range is from 0 to {0} (inclusive)!".format(
                                len(population._agents) - 1))
            else:
                self._saveBestModel(
                        locationForTrainingResults,
                        population,
                        whichModelIsTheBest)
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
                    "type(whichModelIsTheBest) == {1}, " \
                    "type(shouldSavePopulation) == {2}".format(
                            type(population),
                            type(whichModelIsTheBest),
                            type(shouldSavePopulation)))
        
        self._trainingLog.Save(locationForTrainingResults)
    
    def LoadBestModel(self, dirNameWithModelToLoad):
        bestModel = None
        basePathToModel = self._createBasePathForResults()
        fullPathToModel = \
                os.path.join(
                        basePathToModel,
                        dirNameWithModelToLoad,
                        "best_model.pth")
        if os.path.isfile(fullPathToModel):
            bestModel = torch.load(fullPathToModel)

        return bestModel
    
    def _doParametersHaveValidTypes(
            self,
            population,
            whichModelIsTheBest,
            shouldSavePopulation):
        return type(population) == AgentsPopulation \
                and type(whichModelIsTheBest) == int \
                and type(shouldSavePopulation) == bool
    
    def _createLocationForTrainingResults(self):
        basePath = self._createBasePathForResults()
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

    def _saveBestModel(self, location, population, whichModelIsTheBest):
        bestModel = population._agents[whichModelIsTheBest]
        bestModelFileName = "best_model.pth"
        fullPathForModelFile = os.path.join(location, bestModelFileName)
        torch.save(bestModel, fullPathForModelFile)

    def _saveWholePopulation(self, location, population):
        dirForPopulation = "population"
        locationForPopulationFiles = os.path.join(location, dirForPopulation)
        
        if os.path.isdir(locationForPopulationFiles):
            rmtree(locationForPopulationFiles)
        os.mkdir(locationForPopulationFiles)
        
        for i in range(len(population._agents)):
            fileNameForModel = "model_{0}.pth".format(i+1)
            fullPathForModel = \
                    os.path.join(locationForPopulationFiles, fileNameForModel)
            torch.save(population._agents[i], fullPathForModel)
