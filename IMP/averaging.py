import networkx as nx
import networkit as nk
import os
import random
from decimal import Decimal
import networkx.algorithms.community as nx_comm
from IMP import twitter_loc
import matplotlib.pyplot as plt
import math
from statistics import mean
import time

import numpy as np
#import pandas as pd
from IMP.Simulation import *
from IMP import fb_loc


abs_path = os.path.abspath(os.path.dirname(__file__))


# counter measure IDs
COUNTER_MEASURE_NONE = 0
COUNTER_MEASURE_COUNTER_RUMOR_SPREAD = 1
COUNTER_MEASURE_HEAR_FROM_AT_LEAST_TWO = 2
COUNTER_MEASURE_DELAYED_SPREADING = 3
COUNTER_MEASURE_COMMUNITY_DETECTION = 4
COUNTER_MEASURE_DOUBT_SPREADING = 5
COUNTER_MEASURE_TRIANGLE = 6
COUNTER_MEASURE_FACTCHECKING = 7
# counter measure IDs
# node color IDs
NODE_COLOR_RED = 1
NODE_COLOR_WHITE = -1
NODE_COLOR_GRAY = 0
NODE_COLOR_RESERVED = 2
NODE_COLOR_GREEN = 3





def BigSimulation(num_runs, graph_loc, type_graph, num_red,k, dict_args, dict_counter_measure):
    our_graph = GetGraph(graph_loc=graph_loc, type_graph=type_graph, dict_args=dict_args)
    averages = averaging(num_runs=num_runs, our_graph=our_graph,num_red=num_red, k=k, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
    return averages



def averaging(num_runs,our_graph,num_red, k, dict_args, dict_counter_measure):
    listoflists= []
    for i in range(num_runs):
        graylist, n = Simulation(our_graph=our_graph, num_red=num_red, k=k, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
        listoflists.append(graylist)

    #find the list of maximum length
    maxlen = len(listoflists[0])
    for i in range(1,len(listoflists)):
        if len(listoflists[i]) > maxlen:
            maxlen = len(listoflists[i])

    #padd all the lists to the max length and padd with the last value
    padded_listoflists = []
    for i in listoflists:
        padding = [i[-1]]*(maxlen - len(i))
        i.extend(padding)
        padlist = np.array(i)
        padded_listoflists.append(padlist)
    #do elementwise averaging
    averages = list(np.mean(padded_listoflists, axis=0)/n)
    print(averages)

    return averages





