from train_de import train_de
from train_pso import train_pso
from run import run
from src.experiment.ExperimentDataCollector import *
from src.experiment.ChartsGenerator import *
from src.Logger import *
from src.training.TrainingResultsRepository import *
from src.training.training_utilities import loadConfigData
from datetime import datetime
from docopt import docopt
import os
from shutil import rmtree

def getProgramOptions():
    APP_USAGE_DESCRIPTION = """
Run series of experiments which result in generating charts for IT Engineering Thesis.
NOTE: As a config file should be used 'config.json' file or other with appropriate fields.

Usage:
    experiment.py <config-file-path> [options]
    experiment.py -h | --help

Options:
    --num-of-trials=<n>         Specify number of trials used to generate data [default: 10].
    -v --verbose                Run in verbose mode
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    return options

def generateScriptOptions(
        envPath,
        configFilePath,
        trackNum,
        isVerbose=False,
        model = None,
        shouldSavePopulation = False,
        pretrainedPopulation = None):
    scriptOptions = {
        "--env-path" : envPath,
        "<config-file-path>" : configFilePath,
        "--track-1" : False,
        "--track-2" : False,
        "--track-3" : False,
        "--model" : model,
        "--verbose" : isVerbose,
        "--save-population" : shouldSavePopulation,
        "--population" : pretrainedPopulation
    }
    scriptOptions["--track-{0}".format(trackNum)] = True
    return scriptOptions

def generateDataFromTraining(
        trainFunction,
        trainingAlgorithm,
        trackNumber,
        pathToConfigFile,
        buildPaths,
        experimentLog,
        dataCollector,
        minFitnessDict,
        isVerbose = False):
    trainTrackName = "RaceTrack_{0}".format(trackNumber)
    envBuildPath = buildPaths[trainTrackName]
    
    # --- Train new model --- #
    trainOptions = generateScriptOptions(
            envBuildPath,
            pathToConfigFile,
            trackNumber,
            isVerbose = isVerbose)
    
    trainFunction(trainOptions, experimentLog, dataCollector)
    
    pathToLastSavedModel = dataCollector.PathToLastSavedModel
    baseDirOfLastTrainedModel = \
            os.path.basename(os.path.normpath(pathToLastSavedModel))
    
    # --- Validate model --- #
    validateTrainedModel(trainingAlgorithm, trackNumber, experimentLog,
            minFitnessDict, buildPaths, baseDirOfLastTrainedModel, dataCollector)
    
    # --- Remove model --- #
    rmtree(pathToLastSavedModel) # Zakomentuj jesli trzeba bedzie debugowac
    dataCollector.PathToLastSavedModel = None

def validateTrainedModel(
        trainingAlgorithm,
        trainTrackNumber,
        experimentLog,
        minFitnessDict,
        buildPaths,
        modelDir,
        dataCollector):
    for runTrackNumber in range(1, 4):
        runTrackName = "RaceTrack_{0}".format(runTrackNumber)
        minimalFitness = minFitnessDict[runTrackName]
        
        runOptions = generateScriptOptions(
                buildPaths[runTrackName],
                "",
                runTrackNumber,
                model = modelDir)
        
        isTrainedModelValidOnTrack = \
                run(runOptions, experimentLog, minimalFitness)
        
        if isTrainedModelValidOnTrack:
            dataCollector.IncrementValidationMatrixEntry(
                    trainingAlgorithm,
                    trainTrackNumber,
                    runTrackNumber)
        
        experimentLog.Append(
                "Validation of trained model: algorithm = '{0}', " \
                "trainTrackNumber = {1}, runTrackNumber = {2}, " \
                "validationResult = {3}".format(
                        trainingAlgorithm,
                        trainTrackNumber,
                        runTrackNumber,
                        isTrainedModelValidOnTrack))

def createPathForCurrentResults(pathToExperimentResultsDir):
    currentDateTime = datetime.now()
    baseDirForCurrentResults = "{0}_{1}_{2}_{3}_{4}_{5}".format(
            str(currentDateTime.year).zfill(2),
            str(currentDateTime.month).zfill(2),
            str(currentDateTime.day).zfill(2),
            str(currentDateTime.hour).zfill(2),
            str(currentDateTime.minute).zfill(2),
            str(currentDateTime.second).zfill(2)
    )
    fullPathToCurrentResults = \
            os.path.join(pathToExperimentResultsDir, baseDirForCurrentResults)
    return fullPathToCurrentResults

def experiment(options):
    # --- Check if directory with Unity builds does exist --- #
    pathToBuildsDir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "env_builds")
    
    if not os.path.isdir(pathToBuildsDir):
        print("Error: 'env_builds' directory doesn't exist! " \
                "Run 'make_builds.py' to fixed it!")
        exit()
    
    # --- Validation of command line args --- #
    numberOfTrials = options["--num-of-trials"]
    errorMessage = "Invalid numberOfTrials value! Should be positive integer! " \
            "Actual type of numberOfTrials: '{0}', actual value: '{1}'.".format(
                    type(numberOfTrials),
                    numberOfTrials)
    
    if isinstance(numberOfTrials, str) and numberOfTrials.isdigit():
        numberOfTrials = int(numberOfTrials)
        if numberOfTrials <= 0:
            print(errorMessage)
            exit()
    else:
        print(errorMessage)
        exit()
    
    del errorMessage

    # --- Create logger object --- #
    experimentLog = \
            Logger(isVerbose = options["--verbose"], fileName = "experiment")
    experimentLog.Append("Experiment log has been created!")
    experimentLog.Append("numberOfTrials = {0}".format(numberOfTrials))
    
    # --- Load config data from file --- #
    pathToConfigFile = options["<config-file-path>"]
    CONFIG_DATA = loadConfigData(pathToConfigFile)
    experimentLog.Append("Config data has been loaded from file: {0}".format(
            pathToConfigFile))
    
    # --- Determine builds paths --- #
    buildTarget = CONFIG_DATA["MakeBuilds"]["Target"]
    buildPaths = CONFIG_DATA["BuildPaths"][buildTarget]
    
    # --- Create data collector object --- #
    dataCollector = ExperimentDataCollector()
    experimentLog.Append("Data collector object has been created!")
    
    # --- Prepare minimal fitness dict for validation purposes --- #
    minFitnessDict = CONFIG_DATA["TrainingParameters"]["minimalAcceptableFitness"]
    
    # --- Experiment sequence loop (TODO) --- #
    for trialCounter in range(numberOfTrials):
        for trackNumber in range(1, 4):
            experimentLog.Append("Generating data from 'train_de.py', track: " \
                    "{0}, trial: {1}".format(trackNumber, trialCounter + 1))
            generateDataFromTraining(
                    train_de,
                    "DE",
                    trackNumber,
                    pathToConfigFile,
                    buildPaths,
                    experimentLog,
                    dataCollector,
                    minFitnessDict,
                    isVerbose = options["--verbose"])
            
            experimentLog.Append("Generating data from 'train_pso.py', track: " \
                    "{0}, trial: {1}".format(trackNumber, trialCounter + 1))
            generateDataFromTraining(
                    train_pso,
                    "PSO",
                    trackNumber,
                    pathToConfigFile,
                    buildPaths,
                    experimentLog,
                    dataCollector,
                    minFitnessDict,
                    isVerbose = options["--verbose"])
    
    # --- Create location for charts --- #
    pathToResultsDir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "experiment_results")
    if not os.path.isdir(pathToResultsDir):
        os.mkdir(pathToResultsDir)
        experimentLog.Append(
                "'{0}' directory has been created!".format(pathToResultsDir))
    
    pathToCurrentResults = createPathForCurrentResults(pathToResultsDir)
    if os.path.isdir(pathToCurrentResults):
        rmtree(pathToCurrentResults)
    
    os.mkdir(pathToCurrentResults)
    experimentLog.Append(
            "'{0}' directory has been created!".format(pathToCurrentResults))
    
    # --- Generate charts --- #
    chartsGenerator = ChartsGenerator(dataCollector, pathToCurrentResults)
    experimentLog.Append("Charts generator objects has been created!")
    chartsGenerator.CreateAll()
    experimentLog.Append("All charts have been generated!")
    
    # --- Save log file --- #
    experimentLog.Append(
            "End of log file, save to '{0}'.".format(pathToCurrentResults))
    experimentLog.Save(pathToCurrentResults)

if __name__ == "__main__":
    options = getProgramOptions()
    experiment(options)
