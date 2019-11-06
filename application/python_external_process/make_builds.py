from docopt import docopt
from src.training.training_utilities import loadConfigData
import os
import subprocess
from shutil import rmtree

def getProgramOptions():
    APP_USAGE_DESCRIPTION = """
Create Unity environment builds. Each build has exactly one scene (race track).
NOTE: As a config file should be used 'config.json' file or other with appropriate fields.
Before run this script, ensure that 'MakeBuilds' section from config file contains valid data.
On 'Unity' field you have to type the path to your Unity editor.
On 'Target' field, you have to type either 'Linux' or 'Windows' (it depends on what OS you have).

Usage:
    make_builds.py <config-file-path>
    make_builds.py -h | --help
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    return options

def make_builds(options):
    pathToConfigFile = options["<config-file-path>"]
    CONFIG_DATA = loadConfigData(pathToConfigFile)
    if CONFIG_DATA == {}:
        print("Wrong path to config file! (config file path = '{0}')" \
                .format(pathToConfigFile))
        exit()
    CONFIG_DATA = CONFIG_DATA["MakeBuilds"]
    
    pathToUnityEditor = CONFIG_DATA["Unity"]
    if not os.path.isfile(pathToUnityEditor):
        print("Error: Wrong path to Unity editor - '{0}' doesn't exist!".format(
                pathToUnityEditor))
        exit()
    
    allowedTargets = ("Linux", "Windows")
    targetName = CONFIG_DATA["Target"]
    if targetName not in allowedTargets:
        print("Error: Wrong target name! Allowed targets: {0}, specified " \
                "target: '{1}'.".format(allowedTargets, targetName))
        exit()
    del allowedTargets
    
    projectRootPath = \
            os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    envBuildsRootPath = \
            os.path.join(projectRootPath, "python_external_process/env_builds")
    
    if os.path.isdir(envBuildsRootPath):
        rmtree(envBuildsRootPath)
    
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
        
    os.remove(pathToTempLogFile)


if __name__ == "__main__":
    options = getProgramOptions()
    make_builds(options)
