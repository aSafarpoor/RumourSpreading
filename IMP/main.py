import sys
import os
from collections import Counter

import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from IMP.Simulation import *

if __name__ == "__main__":
    # sample dict_args for LFR networks

    # sample dict_args for K-CLique networks
    dict_args = {"num_cliques": 10, "clique_size": 100}

    # sample dict_args for moderatelyExpander
    dict_args = {"degree_of_each_supernode": 4, "number_of_supernodes": 40, "nodes_in_clique": 50}

    # sample dict_args for cyclic/complete networks
    dict_args = {"n": 2000}

    dict_counter_measure = {"id": COUNTER_MEASURE_COUNTER_RUMOR_SPREAD, "start_time": 1, "num_green": 10}

    dict_args = {"n": 1000}

    dict_args = {"n": 1000, "tau1": 3, "tau2": 2, "mu": 0.4, "average_degree": 50, "min_degree": 10,
                 "max_degree": 500,
                 "min_community": 150, "max_community": 200, "tol": 0.04, "max_iters": 1000, "seed": None}
    [list_num_white, list_num_black, list_num_gray, list_num_green] = \
        Simulation(graph="none", SNtype=False, type_graph="LFR", num_black=100, gray_p=0,
                   tresh=0.10, d=0, k=5, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
    plt.clf()
    plt.plot(list_num_white, "blue", label="white")
    plt.plot(list_num_black, "black", label="black")
    plt.plot(list_num_gray, "gray", label="gray")
    plt.plot(list_num_green, "green", label="green")
    plt.legend()
    plt.show()
