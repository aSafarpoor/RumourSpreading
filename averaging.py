
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
# counter measure IDs
# node color IDs
NODE_COLOR_RED = 1
NODE_COLOR_WHITE = -1
NODE_COLOR_GRAY = 0
NODE_COLOR_RESERVED = 2
NODE_COLOR_GREEN = 3





def BigSimulation(num_runs, graph_loc, type_graph, num_red,k, dict_args, dict_counter_measure):
    our_graph = GetGraph(graph_loc=graph_loc, type_graph=type_graph, dict_args=dict_args)
    averages = averaging(num_runs=num_runs, our_graph=our_graph, type_graph=type_graph,num_red=num_red, k=k, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
    return averages



def averaging(num_runs,our_graph, type_graph, num_red, k, dict_args, dict_counter_measure):
    listoflists= []
    for i in range(num_runs):
        graylist, n = Simulation_Charlotte(our_graph=our_graph, type_graph=type_graph, num_red=num_red, k=k, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
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









