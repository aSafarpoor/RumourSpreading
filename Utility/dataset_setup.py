"""
These are the main datasets that we use to run out experiments.
The data structure is a dictionary with the following fields:
loc: a string value showing the location of the dataset
directed: a boolean value which shows whether the dataset is directed or not
type: an integer number representing the type of the graph
"""
# imports (START)
import os

# imports (END)
# Dataset type constants (START)
TYPE_SOCIAL_NETWORK = 0
TYPE_ERDOS_RENYI = 1
TYPE_BA = 2
TYPE_LFR = 3
TYPE_HRG = 4
TYPE_MODERATELY_EXPANDER = 5
TYPE_D_REGULAR_RANDOM_GRAPH = 6
TYPE_RING_OF_CLIQUES = 7
TYPE_CYCLIC = 8
TYPE_COMPLETE = 9
# Dataset type constants (END)

# Datasets (START)
# Constant referring to the directory of the code (START)
ABS_PATH_UTILITY = os.path.abspath(os.path.dirname(__file__))
# Constant referring to the directory of the code (END)

facebook = {"loc": ABS_PATH_UTILITY + "/Datasets/facebook_combined.txt", "name": "facebook", "directed": False,
            "type": TYPE_SOCIAL_NETWORK}
twitter = {"loc": ABS_PATH_UTILITY + "/Datasets/twitter_combined", "name": "twitter", "directed": True,
           "type": TYPE_SOCIAL_NETWORK}
slashdot = {"loc": ABS_PATH_UTILITY + "/Datasets/Slashdot0902.txt", "name": "slashdot", "directed": True,
            "type": TYPE_SOCIAL_NETWORK}
eu_email = {"loc": ABS_PATH_UTILITY + "/Datasets/email-Eu-core.txt", "name": "eu_email", "directed": True,
            "type": TYPE_SOCIAL_NETWORK}
pokec = {"loc": ABS_PATH_UTILITY + "/Datasets/soc-pokec-relationships.txt", "name": "Pokec", "directed": True,
         "type": TYPE_SOCIAL_NETWORK}
# Datasets (END)
