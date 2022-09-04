import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from IMP import twitter_loc
from IMP.Simulation import Simulation

if __name__ == "__main__":
    dict_args = {"n": 1000, "tau1": 1.2, "tau2": 2.1, "mu": 0.3, "average_degree": 5.5, "min_degree": 1,
                 "max_degree": 10,
                 "min_community": 20, "max_community": 200, "tol": 0.4, "max_iters": 100, "seed": None}

    # Simulation(graph="whatev", SNtype=False, type_graph="LFR", p=0.1, gray_p=0.1, k=4, c=20, tresh=0.10,
    #            d=1, j=4, dict_args=dict_args)

    Simulation(graph="whatev", SNtype=False, type_graph="LFR", p=1/50000, gray_p=0, k=5, c=10000, tresh=0.10,
               d=0, j=5,dict_args=dict_args)