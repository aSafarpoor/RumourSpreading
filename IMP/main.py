import math
import sys
import os


import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from IMP.Simulation import *
if __name__ == "__main__":
    # sample dict_args for LFR networks



    # sample dict_args for K-CLique networks
    dict_args = {"num_cliques": 10, "clique_size": 100}



    # sample dict_args for cyclic/complete networks
    dict_args = {"n": 2000}


    dict_args = {"n": 1000}



    dict_counter_measure = {"id": COUNTER_MEASURE_COUNTER_RUMOR_SPREAD, "start_time": 5, "num_green": 50}
    dict_counter_measure = {"id": COUNTER_MEASURE_HEAR_FROM_AT_LEAST_TWO}
    dict_counter_measure = {"id": COUNTER_MEASURE_DELAYED_SPREADING, "sleep_timer": 2}



    dict_counter_measure = {"id": COUNTER_MEASURE_COMMUNITY_DETECTION, "threshold_detection": 0.1,
                            "threshold_block": 0.1}
    dict_counter_measure = {"id": COUNTER_MEASURE_NONE}
    dict_args = {"degree_of_each_supernode": 4, "number_of_supernodes": 40, "nodes_in_clique": 50}

    dict_args = {"n": 1500, "tau1": 3, "tau2": 3, "mu": 0.4, "average_degree": 50,
                 "min_community": 200, "tol": 0.1, "max_iters": 1000, "seed": 7}


    dict_counter_measure = {"id": COUNTER_MEASURE_DOUBT_SPREADING,"negative_doubt_shift":-0.6,"positive_doubt_shift":0.1}
    [list_num_white, list_num_black, list_num_gray, list_num_green] = \
        Simulation(graph="none", SNtype=False, type_graph="LFR", num_black=100, gray_p=0,
                   tresh=0.10, d=0, k=5, dict_args=dict_args, dict_counter_measure=dict_counter_measure,seed=9)

    # plot the results
    plt.clf()
    plt.plot(list_num_white, "blue", label="white")
    plt.plot(list_num_black, "black", label="black")
    plt.plot(list_num_gray, "gray", label="gray")
    plt.plot(list_num_green, "green", label="green")
    plt.legend()
    plt.show()


