import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from IMP import twitter_loc
from IMP.Simulation import Simulation

if __name__ == "__main__":

    # sample dict_args for LFR networks
    dict_args = {"n": 1000, "tau1": 1.2, "tau2": 2.1, "mu": 0.3, "average_degree": 5.5, "min_degree": 1,
                 "max_degree": 10,
                 "min_community": 20, "max_community": 200, "tol": 0.4, "max_iters": 100, "seed": None}

    # sample dict_args for K-CLique networks
    dict_args = {"num_cliques":10,"clique_size":100}

    # sample dict_args for moderatelyExpander
    dict_args = {"degree_of_each_supernode":3,"number_of_supernodes":4,"nodes_in_clique":5}

    # Simulation(graph="whatev", SNtype=False, type_graph="LFR", p=0.1, gray_p=0.1, k=4, c=20, tresh=0.10,
    #            d=1, j=4, dict_args=dict_args)
    Simulation(graph="whatev", SNtype=False, type_graph="moderatelyExpander", p=1/1000, gray_p=0, tresh=0.10,
               d=0,k=5, dict_args=dict_args)