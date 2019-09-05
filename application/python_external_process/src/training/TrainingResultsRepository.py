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

    # TODO - dorobić zabezpieczenia + odpowiednie testy które to weryfikują
    def Save(self, population, whichModelIsTheBest, shouldSavePopulation):
        locationForTrainingResults = self._createLocationForTrainingResults()
        os.mkdir(locationForTrainingResults)
        self._saveBestModel(
                locationForTrainingResults,
                population,
                whichModelIsTheBest)
        if shouldSavePopulation:
            self._saveWholePopulation(locationForTrainingResults, population)
        self._trainingLog.Save(locationForTrainingResults)
    
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
            
        
        
