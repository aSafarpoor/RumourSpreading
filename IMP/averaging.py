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

#charlotte's path
abs_path = os.path.abspath(os.path.dirname(__file__))
#sajjad's path
#datasets_path = os.path.join(os.path.abspath(""), "Datasets")

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



def averaging_acorss_experiments(listoflists):
    #find the list of maximum length
    maxlen = len(listoflists[0])
    for i in range(1, len(listoflists)):
        if len(listoflists[i]) > maxlen:
            maxlen = len(listoflists[i])

    # padd all the lists to the max length and padd with the last value
    padded_listoflists = []
    for i in listoflists:
        padding = [i[-1]] * (maxlen - len(i))
        i.extend(padding)
        print(i)
        padded_listoflists.append(i)
    #print(padded_listoflists)
    #print(len(padded_listoflists))

    return padded_listoflists

def averaging2(listoflists):
    # find the list of maximum length
    n = 81306
    maxlen = len(listoflists[0])
    for i in range(1, len(listoflists)):
        if len(listoflists[i]) > maxlen:
            maxlen = len(listoflists[i])

    # padd all the lists to the max length and padd with the last value
    padded_listoflists = []
    for i in listoflists:
        padding = [i[-1]] * (maxlen - len(i))
        i.extend(padding)
        padlist = np.array(i)
        padded_listoflists.append(padlist)
    # do elementwise averaging
    averages = list(np.mean(padded_listoflists, axis=0) / n)
    print(averages)

    return averages

twitter_CM2_list = [[0, 0, 0, 0, 0, 1, 8, 26, 130, 637, 2666, 8248, 17223, 26682, 33503, 38597, 42059, 44260, 45610, 46506, 47100, 47502, 47754, 47908, 48032, 48123, 48203, 48355, 48423, 48480, 48601, 48660, 48684, 48696, 48707, 48713, 48720, 48728, 48734, 48740, 48748, 48755, 48758, 48759, 48760, 48760, 48760, 48760, 48761, 48761, 48761, 48761, 48761, 48761, 48761],
                    [0, 0, 0, 0, 0, 1, 2, 4, 5, 5, 5, 6, 7, 7, 9, 13, 20, 53, 145, 474, 1499, 2950, 5700, 11989, 21497, 30959, 37538, 41749, 44291, 45859, 47093, 47856, 48256, 48575, 48850, 49047, 49200, 49360, 49470, 49525, 49559, 49572, 49580, 49588, 49594, 49594, 49595, 49599, 49608, 49635, 49660, 49684, 49705, 49727, 49747, 49775, 49800, 49812, 49820, 49823, 49826, 49830, 49833, 49834, 49834, 49834, 49834],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 5, 15, 124, 955, 4587, 11342, 19756, 26715, 32731, 37227, 40409,
                     42485, 43989, 44975, 45584, 45970, 46229, 46391, 46521, 46655, 46833, 46937, 46996, 47039, 47064,
                     47073, 47090, 47131, 47169, 47187, 47197, 47202, 47204, 47204, 47204, 47204, 47204, 47204, 47204],
                    [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2],
                    [0, 0, 0, 0, 0, 1, 2, 8, 64, 236, 850, 2702, 6794, 14172, 24078, 32379, 37697, 41230, 43495, 44925, 45857, 46475, 46852, 47114, 47285, 47421, 47568, 47706, 47861, 48017, 48104, 48193, 48240, 48265, 48276, 48279, 48284, 48285, 48286, 48286, 48286, 48286, 48286],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2],
                    [0, 0, 0, 0, 0, 1, 5, 58, 200, 556, 1620, 4157, 8099, 13528, 21599, 29700, 35467, 39545, 42331, 43984, 44988, 45650, 46075, 46298, 46455, 46543, 46635, 46724, 46778, 46821, 46847, 46873, 46889, 46908, 46916, 46932, 46942, 46952, 46963, 46983, 47018, 47037, 47043, 47045, 47047, 47048, 47048, 47048, 47049, 47050, 47050, 47050, 47050, 47050, 47050],
                    [0, 0, 0, 0, 0, 1, 2, 4, 11, 19, 36, 69, 123, 252, 517, 1684, 6007, 13396, 22372, 30372, 36209, 39887, 42213, 43779, 44815, 45418, 45809, 46070, 46236, 46386, 46478, 46565, 46634, 46704, 46821, 46931, 47005, 47031, 47040, 47048, 47051, 47051, 47051],
                    [0, 0, 0, 0, 0, 1, 1]
                    ]

averaging2(twitter_CM2_list)
