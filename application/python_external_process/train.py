""" STEP 0: Import Required Packages """
# from mlagents.envs import UnityEnvironment
# from src.genetic.AgentsPopulation import *
# from src.genetic.GeneticAlgorithm import *
from docopt import docopt

########################## Szkic podziału kodu na części (funkcje, moduły) ##########################
def load_config():
    pass

def prepare_training_parameters():
    pass

def run_training():
    pass

def main():
    APP_USAGE_DESCRIPTION = """
Train neural networks to drive a car on a racetrack. Racetrack must be valid Unity ML-Agents environment.

Usage:
    train.py <config-file-path> (--genetic | --neat | --ppo) [options]
    train.py -h | --help

Options:
    -v --verbose                    Run in verbose mode
    --genetic                       Use Genetic Algorithm to learn network
    --neat                          Use NEAT algorithm to learn network
    --ppo                           Use PPO algorithm to learn network
    --model=<pretrained-model>      Specify path to pretrained network
    --env-path=<unity-build>        Specify path to Unity environment build
"""
    options = docopt(APP_USAGE_DESCRIPTION)
    print("Config: %s" % options["<config-file-path>"])
    if options["--neat"]:
        print("neat")
    elif options["--genetic"]:
        print("genetic")
    elif options["--ppo"]:
        print("ppo")
    if options["--verbose"]:
        print("verbose output")
    print("Model: %s" % options["--model"])
    print("Env_path: %s" % options["--env-path"])
    
if __name__ == "__main__":
    main()

# ############################# Kod właściwy (na brudno) ###########################################
# """ STEP 1: Set the Training Parameters
# ======
        # TODO: ogarnij co ma tu być
# """
# # Trenuj najwyżej do tyle epizodów
# MAX_EPISODES_COUNT = ???
# # Możesz przerwać dalszy trening, jeśli osiągniesz ten pułap przystosowania
# MINIMAL_ACCEPTABLE_FITNESS = ???
# # Topologia sieci (ile neuronów w każdej warstwie)
# HIDDEN_LAYERS_DIMENSIONS = ???
# # Ile maks. epizodów bez rozwoju populacji, zanim przerwiesz trening
# MAX_EPISODES_WITHOUT_IMPROVEMENT = ???
# # Ile procent populacji ma zostać wyselekcjonowane do reprodukcji
# SELECTION_PERCENT_RATE = 10
# # Liczba float od 0 do 1
# MIN_PROBABILITY_TO_MUTATE_CHROMOSOME = 0.4
# # Liczba float od 0 do 1
# MIN_PROBABILITY_TO_MUTATE_GENOME = 0.7
# # Maksymalna liczba kroków na epizod
# MAX_STEPS_PER_EPISODE = ???
# # List for fitness values
# fitnessList = []

# """ STEP 2: Start the Unity Environment
    # Use the "file_name=<path_to_build>" parameter to train from build
    # Empty string means using Unity editor directly
# """
# env = UnityEnvironment(file_name="")

# """ STEP 3: Get The Unity Environment Brain
# Unity ML-Agent applications or Environments contain "BRAINS" which are responsible for deciding 
# the actions an agent or set of agents should take given a current set of environment (state) observations.

# My environment has only one Brain, thus, I just need to access the first brain available (i.e., the default brain).
# I then set the default brain as the brain that will be controlled.
    # -> HOWEVER: There will be multiple agents connected to that brain,
    # and each agent will have its own neural network (because of evolutionary algorithms nature).
# """
# # Get the default brain 
# brain_name = env.brain_names[0]
# # Assign the default brain as the brain to be controlled
# brain = env.brains[brain_name]

# """ STEP 4: Determine number of agents. Determine the size of the Action and Observation Spaces.
    # -> The observation space consists of n variables corresponding to the input from car sensors. 
       # Two continuous actions are available, corresponding to throttle (which can be negative) and steering angle.

# My environment will contain multiple agents in the environment. Each agent behaves independently from others.
# Number of agents is specified in Unity editor.
# All Observation variables will be floating point numbers from the range [0:1]
# All Action variables will be floating point numbers from the range [-1:1]
# """
# # Get info about initial state of environment:
# brainInfo = env.reset(train_mode=True)[brain_name]
# # TODO: Get number of agents in Environment
# numberOfAgents = len(brainInfo.agents)
# # TODO: Get size of Observation space
# observationSize = brainInfo.vector_observations.shape[1]
# # TODO: Get size of Action space
# actionSize = brain.vector_action_space_size

# """ STEP 5: Create population of agents
    # -> TODO: ogarnij jak to zrobić
# """
# agentDimensions = [ observationSize ] + HIDDEN_LAYERS_DIMENSIONS + [ actionSize ]
# population = AgentsPopulation(numberOfAgents, agentDimensions)

# """ STEP 6: Run the Training Sequence
    # -> TODO: ogarnij co tu ma być
# """
# for episodeIndex in range(MAX_EPISODES_COUNT):
    # brainInfo = env.reset(train_mode=True)[brain_name]
    
    

# """ STEP 7: Everything is Finished -> Close the Environment. """
# env.close()
# #################################################################################################################################################
