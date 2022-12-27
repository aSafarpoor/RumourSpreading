import math
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)


import matplotlib.pyplot as plt
from IMP import fb_loc, twitter_loc, SD_loc, Pokec_loc, Gplus_loc,  yt_loc, dblp_loc, Epinions_loc, ca_cit_loc, HR_loc, twitch_loc, wiki_loc, reddit_loc

from IMP.Simulation import *
from IMP.averaging import *
if __name__ == "__main__":
    #dict_counter_measure = {'id':COUNTER_MEASURE_COMMUNITY_DETECTION, 'p_orange_h': 0.05, 'p_orange_r':0.1}
    dict_counter_measure = {'id':COUNTER_MEASURE_FACTCHECKING, 'frac_green':0.1}
    dict_args = {"degree_of_supernodes": 150, "number_of_supernodes": 1000, "nodes_in_clique": 16}
    #dict_args = {"num_cliques":1000, "clique_size":16}
    #dict_args = {"num_nodes": 16000, "p": 4/ math.sqrt(16000)}
    #dict_args = {"num_nodes": 16000, "p": 1 / (4*math.sqrt(16000))}
    #dict_args={'gray_p':0.1}
    #dict_args = {}



    list_gray = BigSimulation(num_runs=50, graph_loc=twitter_loc, type_graph='SN', num_red=1, k=5, dict_args=dict_args,dict_counter_measure=dict_counter_measure)
    plt.clf()
    plt.plot(list_gray, "orange", label="orange")
    plt.legend(["SD"], loc=2)
    plt.xlabel(r'rounds', fontsize=15)
    plt.ylabel("fraction of orange nodes", fontsize=15)
    plt.show()
