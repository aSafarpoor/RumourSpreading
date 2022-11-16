"""
The code and instructions for running the simulations can be found in this file.
"""
# imports (START)
import math
import sys
import os
import matplotlib.pyplot as plt
from Simulation import *
from Utility.dataset_setup import facebook, twitter, slashdot, pokec
# from averaging import averaging
# from averaging import *

# imports (END)


# Constant referring to the directory of the code (START)
ABS_PATH = os.path.abspath(os.path.dirname(__file__))
# Constant referring to the directory of the code (END)

if __name__ == "__main__":
    """
    To run experiments on graphs, we use dictionaries like `dict_args_ER_2000`.
    Every graph dictionary has a `type` field which is set based on the rules of dataset types explained in 
    `dataset_setup.py`. The other fields used in these dictionaries can differ based on the type of the graph.
    
    - Example ER graph:
    dict_args_ER_2000 = {"type":TYPE_ERDOS_RENYI,"n": 2000}
    type: indicates the type of the graph (Erdos-Renyi)
    n:  indicates the number of nodes
    
    - Example BA graph
    dict_args_BA_2000_300 = {"type":TYPE_BA,"n": 2000,"m":300}
    type: indicates the type of the graph (Barabási–Albert preferential attachment)
    n:  indicates the number of nodes
    m: indicates the number of edges that are preferentially attached to existing nodes with high degree
    
    - Example D-regular graph
    dict_args_DREG_2000_50 = {"type": TYPE_D_REGULAR_RANDOM_GRAPH, "n": 2000, "d": 50}
    type: indicates the type of the graph (D-regular random graph)
    n:  indicates the number of nodes
    d: the degree of each node
    
    - Example Hyperbolic Random Graph
    dict_args_HRG_2000_10 = {"type": TYPE_HRG, "n": 2000, "avg_degree": 10}
    type: indicates the type of the graph (Hyperbolic random graph)
    n: the number of nodes
    avg_degree: average degree
    
    - Example Cyclic graph
    dict_args_CYCLE_2000 = {"type": TYPE_CYCLIC, "n": 2000}
    type: indicates the type of the graph (cyclic graph)
    n: the number of nodes
     
     
    - Example ring of cliques
    dict_args_TYPE_RING_OF_CLIQUES_1000_16 = {"type": TYPE_RING_OF_CLIQUES,"num_cliques":1000, "clique_size": 16}
    type: indicates the type of the graph (ring of cliques graph)
    num_cliques: the number of cliques
    clique_size: the number of nodes in each clique
    
    - Example Complete graph
    dict_args_TYPE_COMPLETE_1000 = {"type": TYPE_COMPLETE,"n":1000}
    type: indicates the type of the graph (complete graph)
    n: the number of nodes
    
    - Example Moderately expander graph
    dict_args_TYPE_MODERATELY_EXPANDER_1000_50_16 = {"type": TYPE_MODERATELY_EXPANDER, "number_of_supernodes": 1000,
                                                     "degree_of_supernodes": 50, "nodes_in_clique": 16}
    type: indicates the type of the graph (Moderately expander graph_ look at Preliminaries section of the paper)
    number_of_supernodes: the number of supernodes in the graph
    degree_of_supernodes: the degree of each supernode
    nodes_in_supernodes: the number of nodes within each supernode
    
    - Example LFR graph
    dict_args_LFR = {"type":TYPE_LFR,"n": 1500, "tau1": 3, "tau2": 3, "mu": 0.4, "average_degree": 50,
                 "min_community": 200, "tol": 0.1, "max_iters": 1000, "seed": 7}
    type: indicates the type of the graph (LFR networks)
    n:  int Number of nodes in the created graph.
    tau1:   float   Power law exponent for the degree distribution of the created graph.
            This value must be strictly greater than one.
    tau2:   float   Power law exponent for the community size distribution in the created graph.
            This value must be strictly greater than one.
    mu: float   Fraction of inter-community edges incident to each node.
        This value must be in the interval [0, 1].
    average_degree: float   Desired average degree of nodes in the created graph.
                    This value must be in the interval [0, n].
    min_degree: int Minimum degree of nodes in the created graph. This value must be in the interval [0, n].
                Exactly one of this and average_degree must be specified, otherwise a NetworkXError is raised.
    max_degree: int Maximum degree of nodes in the created graph.
    min_community:  int Minimum size of communities in the graph.
    max_community:  int Maximum size of communities in the graph.
    tol:    float   Tolerance when comparing floats, specifically when comparing average degree values.
    max_iters:  int Maximum number of iterations to try to create the community sizes, degree distribution,
                and community affiliations.
    seed:   int, random_state, or None (default)    
            Indicator of random number generation state.
    """

    # dataset args (START)
    dict_args_ER_2000 = {"type": TYPE_ERDOS_RENYI, "n": 2000, "p": 0.1}
    dict_args_BA_2000_300 = {"type": TYPE_BA, "n": 2000, "m": 300}
    dict_args_DREG_2000_50 = {"type": TYPE_D_REGULAR_RANDOM_GRAPH, "n": 2000, "d": 50}
    dict_args_HRG_2000_10 = {"type": TYPE_HRG, "n": 2000, "avg_degree": 10}
    dict_args_CYCLE_2000 = {"type": TYPE_CYCLIC, "n": 2000}
    dict_args_RING_OF_CLIQUES_1000_16 = {"type": TYPE_RING_OF_CLIQUES, "num_cliques": 1000, "clique_size": 16}
    dict_args_COMPLETE_1000 = {"type": TYPE_COMPLETE, "n": 1000}
    dict_args_MODERATELY_EXPANDER_1000_50_16 = {"type": TYPE_MODERATELY_EXPANDER, "number_of_supernodes": 1000,
                                                "degree_of_supernodes": 50, "nodes_in_supernodes": 16}
    dict_args_LFR = {"type": TYPE_LFR, "n": 1500, "tau1": 3, "tau2": 3, "mu": 0.4, "average_degree": 50,
                     "min_community": 200, "tol": 0.1, "max_iters": 1000, "seed": 7}
    # dataset args (END)
    """
    To run different counter measures, we define counter measure parameters in a dictionary. 
    Every counter measure dictionary has a `type` field which shows the type of the counter measure. 
    The other fields used in these dictionaries can differ based on the type of the counter measure.
    
    - Sample experiment setup without any counter measures:
    dict_counter_measure = {"type": COUNTER_MEASURE_NONE}
    type: indicates the type of the counter measure (none)

    - Sample experiment setup with green information spreading counter measure:
    dict_counter_measure_green_information = {"type": COUNTER_MEASURE_GREEN_INFORMATION, "start_time": 5,
                                              "num_green": 1,"green_spreading_ratio":0.5,
                                              "high_degree_selection_strategy": False}
    type: indicates the type of the counter measure (green information)
    start_time: the time slot when the green information spreading process starts
    num_green: the number of green nodes in the first round of the counter measure
    green_spreading_ratio: people are less interested in authorized news, so we add another random selection
    high_degree_selection_strategy: True for selecting high degree nodes and False for random selection strategy
    - Sample experiment setup with hearing from at least neighbor counter measure:
    dict_counter_measure_hear_from_two = {"type": COUNTER_MEASURE_HEAR_FROM_AT_LEAST_TWO}
    type: indicates the type of the counter measure (hearing from at least two)
    
    - Sample experiment setup with delayed rumor spreading counter measure:
    dict_counter_measure_delayed_spreading = {"type": COUNTER_MEASURE_DELAYED_SPREADING, "sleep_timer": 2}
    type: indicates the type of the counter measure (delayed spreading)
    sleep_timer: the length of the time slot that each node waits before spreading the rumor
    
    - Sample experiment setup with community detection counter measure:
    dict_counter_measure_community_Detection = {"id": COUNTER_MEASURE_COMMUNITY_DETECTION, "threshold_detection": 0.1,
                            "threshold_block": 0.1}
    type: indicates the type of the counter measure (community detection)
    threshold_detection: the ratio of the all nodes which have to turn red for the counter measure to get activated;
    threshold_block: the ratio of the nodes in a community which have to turn red for the community to get blocked.
    
    - Sample experiment setup with community detection counter measure:
    dict_counter_measure_community_Detection = {"type": COUNTER_MEASURE_COMMUNITY_DETECTION, "threshold_detection": 0.1,
                            "threshold_block": 0.1}
    type: indicates the type of the counter measure (community detection)
    threshold_detection: the ratio of the all nodes which have to turn red for the counter measure to get activated;
    threshold_block: the ratio of the nodes in a community which have to turn red for the community to get blocked.
    
    
    """

    # counter measure args (START)
    dict_counter_measure_none = {"type": COUNTER_MEASURE_NONE}
    dict_counter_measure_green_information = {"type": COUNTER_MEASURE_GREEN_INFORMATION, "start_time": 1,
                                              "num_green": 1,"green_spreading_ratio":0.5,
                                              "high_degree_selection_strategy": True}
    dict_counter_measure_hear_from_two = {"type": COUNTER_MEASURE_HEAR_FROM_AT_LEAST_TWO}
    dict_counter_measure_delayed_spreading = {"type": COUNTER_MEASURE_DELAYED_SPREADING, "sleep_timer": 2}
    dict_counter_measure_community_Detection = {"type": COUNTER_MEASURE_COMMUNITY_DETECTION, "threshold_detection": 0.1,
                                                "threshold_block": 0.1}

    dict_counter_measure_doubt_spreading = {"id": COUNTER_MEASURE_DOUBT_SPREADING, "negative_doubt_shift": -0.6,
                                            "positive_doubt_shift": 0.1}
    # counter measure args (END)

    # running the simulation (START)
    [list_num_white, list_num_red, list_num_orange, list_num_green] = \
        simulation(realworld_graph=facebook, num_red=1, orange_p=0,
                   k=5, dict_args=None, dict_counter_measure=
                   dict_counter_measure_green_information, seed=9)

    # running the simulation (END)
    # plotting and saving the results (START)
    with open("Output/output.txt", "a") as f:
        f.write("list_num_white = " + repr(list_num_white) + "\n")
        f.write("list_num_black = " + repr(list_num_red) + "\n")
        f.write("list_num_gray = " + repr(list_num_orange) + "\n")
        f.write("list_num_green = " + repr(list_num_green) + "\n")
        f.write("------------------------------------------- \n")

    plt.clf()
    plt.xlabel("rounds", fontdict=None, labelpad=None)
    plt.ylabel("the fraction of orange nodes", fontdict=None, labelpad=None)
    plt.plot(list_num_white, "blue", label="white")
    plt.plot(list_num_red, "red", label="red")
    plt.plot(list_num_orange, "orange", label="Orange nodes in "+facebook["name"])
    plt.plot(list_num_green, "green", label="green")
    plt.legend()
    plt.show()
    # plotting and saving the results (END)
