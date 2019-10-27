from docopt import docopt
from src.training.training_utilities import loadConfigData
import os
import subprocess

def make_builds():
    APP_USAGE_DESCRIPTION = """
Create Unity environment builds. Each build has exactly one scene (race track).
NOTE: As a config file should be used 'config/make_config.json' file or other with appropriate fields.

Usage:
    make_builds.py <config-file-path>
    make_builds.py -h | --help
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    
    pathToConfigFile = options["<config-file-path>"]
    CONFIG_DATA = loadConfigData(pathToConfigFile)
    if CONFIG_DATA == {}:
        print("Wrong path to config file! (config file path = '{0}')" \
                .format(pathToConfigFile))
        exit()
    
    pathToUnityEditor = CONFIG_DATA["Unity"]
    targetName = CONFIG_DATA["Target"]
    
    projectRootPath = \
            os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    envBuildsRootPath = \
            os.path.join(projectRootPath, "python_external_process/env_builds")
    
    if os.path.isdir(envBuildsRootPath):
        removeCommand = "rm {0} -rf".format(envBuildsRootPath)
        subprocess.call(args = removeCommand, shell = True)
    
    os.mkdir(envBuildsRootPath)
    for i in range(3):
        pathToSpecifiedRacetrack = \
                os.path.join(envBuildsRootPath, "RaceTrack_{0}".format(i+1))
        os.mkdir(pathToSpecifiedRacetrack)
    
    pathToUnityProject = os.path.join(
            projectRootPath,
            "unity_mla_environment/RacingEnvironments")
    
    pathToTempLogFile = os.path.join(
            projectRootPath,
            "python_external_process/make_builds.log")
    
    commandToMakeBuilds = "{0} -quit -batchmode -logFile {1} -projectPath {2} -executeMethod " \
            "EnvBuilder.Build {3} {4}".format(
                    pathToUnityEditor,
                    pathToTempLogFile,
                    pathToUnityProject,
                    envBuildsRootPath,
                    targetName)
    
    try:
        subprocess.check_call(args = commandToMakeBuilds, shell = True)
    except subprocess.CalledProcessError:
        print("Something goes wrong while creating builds! Please check '{0}' " \
                "for further details!".format(pathToTempLogFile))
        exit()
        
    removeTempLogCommand = "rm {0}".format(pathToTempLogFile)
    subprocess.call(args = removeTempLogCommand, shell = True)


if __name__ == "__main__":
    make_builds()
