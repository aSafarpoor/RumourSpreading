

import numpy as np
import pandas as pd
from IMP.Simulation import Simulation_no_countermeasure
from IMP import fb_loc









def averaging(num_runs,graph, type_graph, num_red, k, dict_args):
    listoflists= []
    for i in range(num_runs):
        graylist, n = Simulation_no_countermeasure(graph, type_graph, num_red, k, dict_args)
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




#listoflists = [[0, 0, 0, 0, 0, 1, 3, 9, 13, 30, 78, 137, 148, 160, 164, 170, 173, 176, 177, 177, 177, 177, 177, 177, 177], [0, 0, 0, 0, 0, 1, 76, 220, 312, 468, 590, 670, 717, 774, 814, 843, 864, 874, 880, 889, 892, 896, 896, 896, 896], [0, 0, 0, 0, 0, 1, 7, 61, 170, 276, 623, 898, 1063, 1307, 1658, 2019, 2291, 2458, 2628, 2785, 2860, 2927, 2963, 2977, 2983, 2986, 2986, 2986, 2986, 2986, 2986]]
#averaging(listoflists=listoflists, num_runs=3, type_graph="SN", graph=fb_loc, num_red=1, dict_args={},k=5)








