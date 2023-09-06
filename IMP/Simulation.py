# imports
from statistics import mean
import os
import random
from decimal import Decimal
import math
import time
import networkx as nx
import networkit as nk
import pandas as pd
import community.community_louvain as cl
# import matplotlib.pyplot as plt


abs_path = os.path.abspath(os.path.dirname(__file__))


# from IMP import fb_loc, twitter_loc,  musea_DE_loc, musea_FR_loc
twitter_loc = '../networks/twitter_combined.txt'
fb_loc = '../networks/facebook_combined.txt'
musea_DE_loc = '../networks/musae_DE_edges.csv'
musea_FR_loc = '../networks/musae_FR_edges.csv'

# counter measure IDs
COUNTER_MEASURE_NONE = 0
COUNTER_MEASURE_GREEN_NODES = 1
COUNTER_MEASURE_HEAR_FROM_AT_LEAST_TWO = 2
COUNTER_MEASURE_DELAYED_SPREADING = 3
COUNTER_MEASURE_COMMUNITY_DETECTION = 4
COUNTER_MEASURE_DOUBT_SPREADING = 5
COUNTER_MEASURE_TRIANGLE = 6
COUNTER_MEASURE_FACTCHECKING = 7
COUNTER_MEASURE_ORANGE_NODES = 8
COUNTER_MEASURE_ACCURACYFLAG = 9
# node color IDs
NODE_COLOR_RED = 1
NODE_COLOR_WHITE = -1
NODE_COLOR_GRAY = 0
NODE_COLOR_RESERVED = 2
NODE_COLOR_GREEN = 3
NODE_COLOR_LG = 4


# node colour IDs
# https://nbviewer.org/gist/anonymous/bb4e1dfafd9e90d5bc3d
def KClique(j, c):
    G = nx.ring_of_cliques(j, c)

    return G

def FlowerGraph(num_cliques, nodes_in_clique):
    G = nx.complete_graph(n=nodes_in_clique)

    for i in range(num_cliques-1):
        G = nx.disjoint_union(G, nx.complete_graph(n=nodes_in_clique))
    total_nodes = nodes_in_clique*num_cliques
    for i in range(0,total_nodes-nodes_in_clique, nodes_in_clique):
        G.add_edge(i, i+nodes_in_clique)
    G.add_edge(0, total_nodes-nodes_in_clique)
    return G

def moderatelyExpander(degree_of_each_supernode, number_of_supernodes, nodes_in_clique):
    H = nx.random_regular_graph(d=degree_of_each_supernode, n=number_of_supernodes)

    G = nx.complete_graph(n=nodes_in_clique)
    H_nodes = list(H.nodes())

    for i in range(len(H_nodes) - 1):
        G = nx.disjoint_union(G, nx.complete_graph(n=nodes_in_clique))
    for i in H_nodes:
        edges_i = list(H.edges(i))
        for j in range(len(edges_i)):
            G.add_edge(
                random.randint(edges_i[j][0] * nodes_in_clique, edges_i[j][0] * nodes_in_clique + nodes_in_clique - 1),
                random.randint(edges_i[j][1] * nodes_in_clique, edges_i[j][1] * nodes_in_clique + nodes_in_clique - 1))
        H.remove_node(i)
    return G

def GetGraph(graph_loc, type_graph, dict_args):
    print('graph loc', graph_loc)
    if graph_loc == musea_DE_loc:
        df = pd.read_csv(os.path.join(abs_path, graph_loc))
        temp_graph = nx.from_pandas_edgelist(df, 'from', 'to')
    elif graph_loc == musea_FR_loc:
        df = pd.read_csv(os.path.join(abs_path, graph_loc))
        temp_graph = nx.from_pandas_edgelist(df, 'from', 'to')
    else:
        temp_graph = nx.read_edgelist(os.path.join(abs_path, graph_loc), create_using=nx.Graph(), nodetype=int)
    n = temp_graph.number_of_nodes()
    print("n",n)
    mapping = dict(zip(temp_graph, range(0, temp_graph.number_of_nodes())))
    temp_graph = nx.relabel_nodes(temp_graph, mapping)

    total = sum(j for i, j in list(temp_graph.degree(temp_graph.nodes)))
    av_deg = total / temp_graph.number_of_nodes()
    p = total / (temp_graph.number_of_nodes() * (temp_graph.number_of_nodes() - 1))
    d = round(n*p/2) *2

    if type_graph == 'ERSN':
        our_graph = nx.fast_gnp_random_graph(n=temp_graph.number_of_nodes(), p=p)
    elif type_graph == 'ER':
        our_graph = nx.fast_gnp_random_graph(n=dict_args['num_nodes'], p=dict_args['p'])
    elif type_graph == 'BA':
        our_graph = nx.barabasi_albert_graph(n=temp_graph.number_of_nodes(), m=int(av_deg))
    elif type_graph == "DREGSN":
        our_graph = nx.random_regular_graph(d=d, n=n)
    elif type_graph == "DREG":
        our_graph = nx.random_regular_graph(d=dict_args['d'], n=dict_args['num_nodes'])
    elif type_graph == "HRG":
        hg = nk.generators.HyperbolicGenerator(n=temp_graph.number_of_nodes(), k=av_deg, gamma=2.5, T=0.6)
        hgG = hg.generate()
        our_graph = nk.nxadapter.nk2nx(hgG)
    elif type_graph == 'SN':
        our_graph = temp_graph

    elif type_graph == "cycle":
        our_graph = nx.cycle_graph(n)
    elif type_graph == "FlowerGraph":
        our_graph = FlowerGraph(dict_args["num_cliques"], dict_args["clique_size"])
    elif type_graph == "Complete":
        our_graph = nx.complete_graph(dict_args["n"])
    elif type_graph == "moderatelyExpander":
        print(dict_args)
        print(dict_args['degree_of_supernodes'])
        print(dict_args['number_of_supernodes'])
        print(dict_args['nodes_in_clique'])
        our_graph = moderatelyExpander(degree_of_each_supernode=dict_args["degree_of_supernodes"],
                                           number_of_supernodes=dict_args["number_of_supernodes"],
                                           nodes_in_clique=dict_args["nodes_in_clique"])

    return our_graph

def GetInitialOpinions(graph, num_red, dict_args, dict_counter_measure):
    cur_num_green_i = 0
    cur_num_gray_r = 0
    cur_num_gray_i = 0
    same_deg_too_many = 0
    if dict_counter_measure['id'] == COUNTER_MEASURE_ORANGE_NODES:
        threshold_orange_nodes = sorted(graph.degree(), key=lambda pair: pair[1], reverse=True)[:int(dict_counter_measure['p_orange_h'] * graph.number_of_nodes())][-1][1]
    if dict_counter_measure['id'] == COUNTER_MEASURE_GREEN_NODES:
        threshold_green_nodes = sorted(graph.degree(), key=lambda pair: pair[1], reverse=True)[:dict_counter_measure['num_green']][-1][1]
    for node in graph.nodes:
        graph.nodes[node]['hit_counter'] = 0
        graph.nodes[node]['stamp'] = 0
        graph.nodes[node]['fc'] = 0
        graph.nodes[node]['temp_vote'] = -5
    if dict_counter_measure['id'] == COUNTER_MEASURE_ORANGE_NODES:
        for (node, degrees) in graph.degree():
            if degrees > threshold_orange_nodes - 1:
                if cur_num_gray_i < int(dict_counter_measure['p_orange_h'] * graph.number_of_nodes()):
                    graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                    cur_num_gray_i = cur_num_gray_i + 1
                else:
                    r = random.random()
                    if r <= dict_counter_measure['p_orange_r']:
                        graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                        cur_num_gray_r = cur_num_gray_r + 1
                    else:
                        graph.nodes[node]['vote'] = NODE_COLOR_WHITE
            else:
                r = random.random()
                if r <= dict_counter_measure['p_orange_r']:
                    graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                    cur_num_gray_r  = cur_num_gray_r + 1
                else:
                    graph.nodes[node]['vote'] = NODE_COLOR_WHITE
        whitenodes = [node for node in graph.nodes if graph.nodes[node]['vote'] == NODE_COLOR_WHITE]
        red_nodes = random.choices(whitenodes, k=num_red)
        for rednode in red_nodes:
            graph.nodes[rednode]['vote'] = NODE_COLOR_RED
        return graph
    if dict_counter_measure['id'] == COUNTER_MEASURE_GREEN_NODES:
        for (node,degrees) in graph.degree():
            if degrees > threshold_green_nodes - 1:
                if cur_num_green_i < dict_counter_measure['num_green']:
                    graph.nodes[node]['vote'] = NODE_COLOR_GREEN
                    cur_num_green_i = cur_num_green_i + 1
                else:
                    print('a same degree to many node')
                    same_deg_too_many = same_deg_too_many + 1
                    graph.nodes[node]['vote'] = NODE_COLOR_WHITE
            else:
                graph.nodes[node]['vote'] = NODE_COLOR_WHITE
        whitenodes = [node for node in graph.nodes if graph.nodes[node]['vote'] == NODE_COLOR_WHITE]
        red_nodes = random.choices(whitenodes, k=num_red)
        for rednode in red_nodes:
            graph.nodes[rednode]['vote'] = NODE_COLOR_RED
        return graph

    else:
        for node in graph.nodes:
            graph.nodes[node]['vote'] = NODE_COLOR_WHITE
        whitenodes = [node for node in graph.nodes if graph.nodes[node]['vote'] == NODE_COLOR_WHITE]
        red_nodes = random.choices(whitenodes, k=num_red)
        for rednode in red_nodes:
            graph.nodes[rednode]['vote'] = NODE_COLOR_RED
        if dict_counter_measure['id'] == COUNTER_MEASURE_FACTCHECKING:
            whitenodes = [node for node in graph.nodes if graph.nodes[node]['vote'] == NODE_COLOR_WHITE]
            factcheckers = random.choices(whitenodes,k=int(dict_counter_measure['frac_green']*graph.number_of_nodes()))
            for factchecker in factcheckers:
                graph.nodes[factchecker]['fc'] = 1
                graph.nodes[factchecker]['vote'] = NODE_COLOR_WHITE

        for edge in graph.edges:
            graph.edges[edge]["blocked"] = 0

        return graph

def Sim_CM5(graph,num_red, k, dict_args, dict_counter_measure):
    our_graph = GetInitialOpinions(graph=graph, num_red=num_red, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
    stop = 0
    phase = 0
    n = our_graph.number_of_nodes()
    cur_num_red = num_red
    cur_num_gray = 0
    cur_num_green = 0
    cur_num_white = our_graph.number_of_nodes() - num_red - cur_num_green
    cur_num_lg = 0
    list_num_gray = [cur_num_gray]
    list_num_red = [cur_num_red]
    list_num_white = [cur_num_white]
    list_num_green = [cur_num_green]
    list_num_lg = [cur_num_lg]




    while stop != 1:
        phase = phase + 1
        change = 0

        for round in range(3*k + 3):

            for node in [node for node in our_graph.nodes if our_graph.nodes[node]['fc'] == 1
                                                             and our_graph.nodes[node]['vote'] == NODE_COLOR_GREEN]:
                neighlist = list(our_graph.adj[node])
                # only consider the white neighbors these are the only ones that can be influenced
                for neigh in [neigh for neigh in neighlist if
                              our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE or
                              our_graph.nodes[neigh]['vote'] == NODE_COLOR_RED]:
                    # manually add the nodes to their own neighborhoods
                    neighset = set(our_graph.adj[node])
                    neighset.add(node)
                    neighsetneigh = set(our_graph.adj[neigh])
                    neighsetneigh.add(neigh)
                    intersection_neigh = neighset.intersection(neighsetneigh)
                    union_neigh = neighset.union(neighsetneigh)
                    jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                    r = (jaccard_sim)
                    rand = random.random()
                    if rand < r:
                        change = change + 1
                        cur_num_green = cur_num_green + 1
                        if our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE:
                            cur_num_white = cur_num_white - 1
                            our_graph.nodes[neigh]['vote'] = NODE_COLOR_GREEN
                        if our_graph.nodes[neigh]['vote'] == NODE_COLOR_RED:
                            cur_num_red = cur_num_red - 1
                            our_graph.nodes[neigh]['vote'] = NODE_COLOR_GREEN

            if (round % 3) == 0:

                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED
                             or (our_graph.nodes[node]['vote'] == NODE_COLOR_GREEN and our_graph.nodes[node]['fc']==0)]:
                    if our_graph.nodes[node]['vote'] == NODE_COLOR_RED:
                        our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1

                        if our_graph.nodes[node]['stamp'] == k:
                            our_graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                            cur_num_gray = cur_num_gray + 1
                            cur_num_red = cur_num_red - 1
                            change = change + 1
                        else:
                            neighlist = list(our_graph.adj[node])
                            # only consider the white neighbors these are the only ones that can be influenced
                            for neigh in [neigh for neigh in neighlist if
                                          our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE]:
                                # manually add the nodes to their own neighborhoods
                                if our_graph.nodes[neigh]['fc'] == 1:
                                    denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                                    r = 1 / denom
                                    rand = random.random()
                                    if rand < r:
                                        our_graph.nodes[neigh]['vote'] = NODE_COLOR_GREEN
                                        cur_num_white = cur_num_white - 1
                                        cur_num_green = cur_num_green + 1
                                        change = change + 1
                                else:
                                    neighset = set(our_graph.adj[node])
                                    neighset.add(node)
                                    neighsetneigh = set(our_graph.adj[neigh])
                                    neighsetneigh.add(neigh)
                                    intersection_neigh = neighset.intersection(neighsetneigh)
                                    union_neigh = neighset.union(neighsetneigh)
                                    jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                                    denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                                    r = (jaccard_sim / denom)
                                    rand = random.random()
                                    if rand < r:
                                        our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                        change = change + 1
                                        cur_num_red = cur_num_red + 1
                                        cur_num_white = cur_num_white - 1
                    if our_graph.nodes[node]['vote'] == NODE_COLOR_GREEN:
                        our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                        if our_graph.nodes[node]['stamp'] == k:
                            our_graph.nodes[node]['vote'] = NODE_COLOR_LG
                            cur_num_green = cur_num_green - 1
                            cur_num_lg = cur_num_lg + 1
                            change = change + 1
                        else:
                            neighlist = list(our_graph.adj[node])
                            for neigh in [neigh for neigh in neighlist if
                                          our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE
                                          or our_graph.nodes[neigh]['vote'] == NODE_COLOR_RED]:
                                neighset = set(our_graph.adj[node])
                                neighset.add(node)
                                neighsetneigh = set(our_graph.adj[neigh])
                                neighsetneigh.add(neigh)
                                intersection_neigh = neighset.intersection(neighsetneigh)
                                union_neigh = neighset.union(neighsetneigh)
                                jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                                denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                                r = (jaccard_sim / denom)
                                rand = random.random()
                                if rand < r:
                                    change = change + 1
                                    cur_num_green = cur_num_green + 1
                                    if our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE:
                                        cur_num_white = cur_num_white - 1
                                        our_graph.nodes[neigh]['vote'] = NODE_COLOR_GREEN
                                    if our_graph.nodes[neigh]['vote'] == NODE_COLOR_RED:
                                        cur_num_red = cur_num_red - 1
                                        our_graph.nodes[neigh]['vote'] = NODE_COLOR_GREEN
                list_num_gray.append(cur_num_gray)
                list_num_lg.append(cur_num_lg)
                list_num_red.append(cur_num_red)
                list_num_green.append(cur_num_green)
                list_num_white.append(cur_num_white)

                print("listnumgray", list_num_gray)
                print("listnumlg", list_num_lg)
                print("listnumred", list_num_red)
                print("listnumgreen",list_num_green)
                print("listnumwhite", list_num_white)
        print("change", change)
        if change == 0:
            stop = 1


    return list_num_gray, n

def Sim_NOCM(our_graph, num_red, k, dict_args, dict_counter_measure):
    print("in noCM code")

    our_graph = GetInitialOpinions(graph=our_graph, num_red=num_red, dict_args=dict_args,
                                   dict_counter_measure=dict_counter_measure)
    # set initial variables
    stop = 0
    phase = 0
    n = our_graph.number_of_nodes()
    count = 0
    cur_num_red = num_red
    cur_num_gray = 0
    cur_num_white = n - cur_num_red
    list_num_gray = [cur_num_gray]
    list_num_red = [cur_num_red]
    list_num_white = [cur_num_white]
    step = 1
    while stop != 1:
        phase = phase + 1
        change = 0
        for round in range(k + 1):
            for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                # the node has been black for k rounds and becomes gray
                if our_graph.nodes[node]['stamp'] == k:
                    our_graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                    cur_num_gray = cur_num_gray + 1
                    cur_num_red = cur_num_red - 1
                    change = change + 1
                else:
                    neighlist = list(our_graph.adj[node])
                    # only consider the white neighbors these are the only ones that can be influenced
                    for neigh in neighlist:
                        if our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE:
                            # manually add the nodes to their own neighborhoods
                            neighset = set(our_graph.adj[node])
                            neighset.add(node)
                            neighsetneigh = set(our_graph.adj[neigh])
                            neighsetneigh.add(neigh)
                            intersection_neigh = neighset.intersection(neighsetneigh)
                            union_neigh = neighset.union(neighsetneigh)
                            jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                            count = count + 1
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = float(jaccard_sim / denom)
                            rand = random.random()
                            if rand < r:
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                change = change + 1
                                cur_num_red = cur_num_red + 1
                                cur_num_white = cur_num_white - 1
                            else:
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_WHITE

            list_num_white.append(cur_num_white)
            list_num_red.append(cur_num_red)
            list_num_gray.append(cur_num_gray)
            print("listnumwhite", list_num_white)
            print("listnumblack", list_num_red)
            print("listnumgray", list_num_gray)
            step = step + 1
            # print("change", change)
        print("change", change)
        if change == 0:
            stop = 1
    return list_num_gray, n

def Sim_CM1(our_graph, num_red, k,dict_args,dict_counter_measure):
    print("in CM1")

    our_graph = GetInitialOpinions(graph=our_graph, num_red=num_red, dict_args=dict_args,
                                   dict_counter_measure=dict_counter_measure)
    #set initial variables
    stop = 0
    phase = 0
    n = our_graph.number_of_nodes()
    count = 0
    cur_num_red = num_red
    initial_num_gray = len([node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_GRAY])
    cur_num_gray = 0
    cur_num_white = n - initial_num_gray - cur_num_red
    list_num_gray = [cur_num_gray]
    list_num_red = [cur_num_red]
    list_num_white = [cur_num_white]
    step = 1

    orange_sanity = 0
    # sanity check green nodes countermeasure
    for node in our_graph.nodes():
        if our_graph.nodes[node]['vote'] == NODE_COLOR_GRAY:
            orange_sanity = orange_sanity + 1


    print("nodes that should be orange", int(dict_counter_measure['p_orange_h'] * n), dict_counter_measure['p_orange_r']*n,
          int(dict_counter_measure['p_orange_h'] * n) + dict_counter_measure['p_orange_r']*n, orange_sanity)
    while stop != 1:
        phase = phase + 1
        change = 0
        for round in range(k + 1):
            for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                # the node has been black for k rounds and becomes gray
                if our_graph.nodes[node]['stamp'] == k:
                    our_graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                    cur_num_gray = cur_num_gray + 1
                    cur_num_red = cur_num_red - 1
                    change = change + 1
                else:
                    neighlist = list(our_graph.adj[node])
                    # only consider the white neighbors these are the only ones that can be influenced
                    for neigh in neighlist:
                        if our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE:
                            # manually add the nodes to their own neighborhoods
                            neighset = set(our_graph.adj[node])
                            neighset.add(node)
                            neighsetneigh = set(our_graph.adj[neigh])
                            neighsetneigh.add(neigh)
                            intersection_neigh = neighset.intersection(neighsetneigh)
                            union_neigh = neighset.union(neighsetneigh)
                            jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                            count = count + 1
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = float(jaccard_sim / denom)
                            rand = random.random()
                            if rand < r:
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                change = change + 1
                                cur_num_red = cur_num_red + 1
                                cur_num_white = cur_num_white - 1
                            else:
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_WHITE

            list_num_white.append(cur_num_white)
            list_num_red.append(cur_num_red)
            list_num_gray.append(cur_num_gray)
            print("listnumwhite", list_num_white)
            print("listnumblack", list_num_red)
            print("listnumgray", list_num_gray)
            step = step + 1
            # print("change", change)
        print("change", change)
        if change == 0:
            stop = 1
    return list_num_gray, n

def Sim_CM2(our_graph, num_red, k, dict_args, dict_counter_measure):
    our_graph = GetInitialOpinions(graph=our_graph, num_red=num_red, dict_args=dict_args,
                                   dict_counter_measure=dict_counter_measure)

    stop = 0
    phase = 0
    n = our_graph.number_of_nodes()
    count = 0
    cur_num_red = num_red
    cur_num_gray = 0
    cur_num_white = n - cur_num_red
    list_num_gray = [cur_num_gray]
    list_num_red = [cur_num_red]
    list_num_white = [cur_num_white]
    threshold_detection = dict_counter_measure["threshold_detection"]
    threshold_block = dict_counter_measure["threshold_block"]

    communities_dict = cl.best_partition(our_graph)
    list_index_coms = list(set(communities_dict.values()))
    listoflistcoms = []
    # now make a list of lists of all nodes in different communities
    for i in range(len(list_index_coms)):
        listoflistcoms.append([])

    for nodetuple in communities_dict:
        listoflistcoms[communities_dict[nodetuple]].append(nodetuple)

    cs = listoflistcoms
    com_index = 0
    sum_blocked = 0
    for nodetuple in communities_dict:
        our_graph.nodes[nodetuple]['comm'] = communities_dict[nodetuple]

    step = 1
    while stop != 1:
        phase = phase + 1
        change = 0
        for round in range(k + 1):
            red_nodes = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]
            maxes = []
            if len(red_nodes) >= int(our_graph.number_of_nodes() * threshold_detection):
                # we are going to potentially block edges made it through the global threshold
                red_ratio_per_community = []
                red_per_community = []
                counter = 0
                for c in cs:
                    red_ratio_per_community.append(0)
                    red_per_community.append(0)
                    for n in c:
                        if our_graph.nodes[n]['vote'] == NODE_COLOR_RED:
                            red_per_community[counter] = red_per_community[counter] + 1
                            red_ratio_per_community[counter] = red_per_community[counter] / len(c)
                            if red_ratio_per_community[counter] >= threshold_block:
                                maxes.append(counter)
                                for n in cs[counter]:
                                    for edge_n in our_graph.edges(n):
                                        if ((edge_n[0] == n and our_graph.nodes[edge_n[1]]["comm"] !=
                                             our_graph.nodes[n]["comm"])
                                                or (edge_n[1] == n and our_graph.nodes[edge_n[0]]["comm"] !=
                                                    our_graph.nodes[n]["comm"])):
                                            our_graph.edges[edge_n]["blocked"] += 1
                                            sum_blocked = sum_blocked + 1
                    counter = counter + 1

            for node in red_nodes:
                our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                if our_graph.nodes[node]['stamp'] == k:
                    our_graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                    cur_num_gray = cur_num_gray + 1
                    cur_num_red = cur_num_red - 1
                else:
                    neighlist = list(our_graph.adj[node])
                    # only consider the white neighbors these are the only ones that can be influenced
                    for neigh in [neigh for neigh in neighlist if our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE]:
                        neighset = set(our_graph.adj[node])
                        neighset.add(node)
                        neighsetneigh = set(our_graph.adj[neigh])
                        neighsetneigh.add(neigh)
                        intersection_neigh = neighset.intersection(neighsetneigh)
                        union_neigh = neighset.union(neighsetneigh)
                        jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                        count = count + 1
                        denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                        r = (jaccard_sim / denom)
                        rand = random.random()
                        if rand < r:
                            if our_graph.nodes[node]["comm"] in maxes or our_graph.nodes[neigh]['comm'] in maxes:
                                if our_graph.nodes[neigh]["comm"] != our_graph.nodes[node]["comm"]:
                                    continue
                            our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                            change = change + 1
                            cur_num_red = cur_num_red + 1
                            cur_num_white = cur_num_white - 1
            list_num_white.append(cur_num_white)
            list_num_red.append(cur_num_red)
            list_num_gray.append(cur_num_gray)
            print("listnumwhite", list_num_white)
            print("listnumblack", list_num_red)
            print("listnumgray", list_num_gray)
            step = step + 1
            # print("change", change)
        print("change", change)
        if change == 0:
            stop = 1
    return list_num_gray, n

def Sim_CM3(our_graph, num_red, k,dict_args,dict_counter_measure):
    our_graph = GetInitialOpinions(graph=our_graph, num_red=num_red, dict_args=dict_args,
                                   dict_counter_measure=dict_counter_measure)
    #set initial variables
    stop = 0
    phase = 0
    n = our_graph.number_of_nodes()
    count = 0
    cur_num_red = num_red
    initial_num_gray = len([node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_GRAY])
    cur_num_gray = 0
    cur_num_white = n - initial_num_gray - cur_num_red
    list_num_gray = [cur_num_gray]
    list_num_red = [cur_num_red]
    list_num_white = [cur_num_white]
    step = 1
    while stop != 1:
        phase = phase + 1
        change = 0
        for round in range(k + 1):
            for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                # the node has been black for k rounds and becomes gray
                if our_graph.nodes[node]['stamp'] == k:
                    our_graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                    cur_num_gray = cur_num_gray + 1
                    cur_num_red = cur_num_red - 1
                    change = change + 1
                else:
                    neighlist = list(our_graph.adj[node])
                    # only consider the white neighbors these are the only ones that can be influenced
                    for neigh in neighlist:
                        if our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE:
                            # manually add the nodes to their own neighborhoods
                            neighset = set(our_graph.adj[node])
                            neighset.add(node)
                            neighsetneigh = set(our_graph.adj[neigh])
                            neighsetneigh.add(neigh)
                            intersection_neigh = neighset.intersection(neighsetneigh)
                            union_neigh = neighset.union(neighsetneigh)
                            jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                            count = count + 1
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = float(jaccard_sim / denom)
                            rand = random.random()
                            if rand < r:
                                coinflip = random.random()
                                if coinflip < dict_counter_measure['p_coinflip']:
                                    our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                    cur_num_white = cur_num_white - 1
                                    cur_num_red = cur_num_red + 1
                                    change= change + 1
                                else:
                                    our_graph.nodes[neigh]['vote'] = NODE_COLOR_GRAY
                                    cur_num_gray = cur_num_gray + 1
                                    cur_num_white = cur_num_white - 1
                                    change = change + 1
                            else:
                                continue

            list_num_white.append(cur_num_white)
            list_num_red.append(cur_num_red)
            list_num_gray.append(cur_num_gray)
            print("listnumwhite", list_num_white)
            print("listnumblack", list_num_red)
            print("listnumgray", list_num_gray)
            step = step + 1
            # print("change", change)
        print("change", change)
        if change == 0:
            stop = 1
    return list_num_gray, n

def Sim_CM4(our_graph, num_red, k, dict_args, dict_counter_measure):
    our_graph = GetInitialOpinions(graph=our_graph, num_red=num_red, dict_args=dict_args,
                                   dict_counter_measure=dict_counter_measure)
    stop = 0
    phase = 0
    n = our_graph.number_of_nodes()
    count = 0

    cur_num_red = num_red
    cur_num_green = dict_counter_measure['num_green']
    cur_num_white = n - cur_num_red - cur_num_green
    cur_num_gray = 0
    cur_num_lg = 0
    prev_num_lg = 0
    prev_num_green = dict_counter_measure['num_green']

    list_num_gray = [cur_num_gray]
    list_num_red = [cur_num_red]
    list_num_white = [cur_num_white]
    list_num_green = [cur_num_green]
    list_num_lg = [cur_num_lg]

    green_sanity = 0
    # sanity check green nodes countermeasure
    for node in our_graph.nodes():
        if our_graph.nodes[node]['vote'] == NODE_COLOR_GREEN:
            green_sanity = green_sanity + 1

    print("green sanity", green_sanity)
    print("green spreading ratio", dict_counter_measure["green_spreading_ratio"])


    # start the updating process
    step = 0
    while stop != 1:
        phase = phase + 1
        change = 0
        for round in range(k + 1):
            new_green = 0
            for node in [node for node in our_graph.nodes if
                         our_graph.nodes[node]['vote'] == NODE_COLOR_RED or our_graph.nodes[node][
                             'vote'] == NODE_COLOR_GREEN]:
                if our_graph.nodes[node]['vote'] == NODE_COLOR_GREEN:
                    if step >= dict_counter_measure["start_time"]:
                        our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                        if our_graph.nodes[node]['stamp'] == k:
                            our_graph.nodes[node]['vote'] = NODE_COLOR_LG
                            cur_num_green = cur_num_green - 1
                            cur_num_lg = cur_num_lg + 1
                            change = change + 1
                        else:
                            neighlist = list(our_graph.adj[node])
                            # only consider the white neighbors these are the only ones that can be influenced
                            for neigh in neighlist:
                                if our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE:
                                    # manually add the nodes to their own neighborhoods
                                    neighset = set(our_graph.adj[node])
                                    neighset.add(node)
                                    neighsetneigh = set(our_graph.adj[neigh])
                                    neighsetneigh.add(neigh)
                                    intersection_neigh = neighset.intersection(neighsetneigh)
                                    union_neigh = neighset.union(neighsetneigh)
                                    jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                                    count = count + 1
                                    #denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                                    denom = 2
                                    r = float(jaccard_sim / denom) * dict_counter_measure[
                                        "green_spreading_ratio"]
                                    rand = random.random()
                                    if rand < r:
                                        our_graph.nodes[neigh]['vote'] = NODE_COLOR_GREEN
                                        change = change + 1
                                        cur_num_green = cur_num_green + 1
                                        cur_num_white = cur_num_white - 1
                                        new_green = new_green + 1
                    else:
                        our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp']

                if our_graph.nodes[node]['vote'] == NODE_COLOR_RED:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    if our_graph.nodes[node]['stamp'] == k:
                        our_graph.nodes[node]['vote'] = NODE_COLOR_LG
                        cur_num_gray = cur_num_gray + 1
                        cur_num_red = cur_num_red - 1
                        change = change + 1
                    else:
                        neighlist = list(our_graph.adj[node])
                        for neigh in neighlist:
                            if our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE:
                                # manually add the nodes to their own neighborhoods
                                neighset = set(our_graph.adj[node])
                                neighset.add(node)
                                neighsetneigh = set(our_graph.adj[neigh])
                                neighsetneigh.add(neigh)
                                intersection_neigh = neighset.intersection(neighsetneigh)
                                union_neigh = neighset.union(neighsetneigh)
                                jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                                count = count + 1
                                denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                                r = float(jaccard_sim / denom)
                                rand = random.random()
                                if rand < r:
                                    our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                    change = change + 1
                                    cur_num_red = cur_num_red + 1
                                    cur_num_white = cur_num_white - 1
                                # print("round", round, "phase", phase, cur_num_gray, cur_num_white, cur_num_red)
            print("total nodes, sum all colors", n, cur_num_white + cur_num_red + cur_num_gray + cur_num_green + cur_num_lg)
            print(cur_num_white, cur_num_red, cur_num_gray, cur_num_green, cur_num_lg)
            delta_num_green = cur_num_green - prev_num_green
            delta_num_lg = cur_num_lg - prev_num_lg
            print("new green", new_green)
            print("delta num green", delta_num_green)
            print('delta num lg', delta_num_lg)
            prev_num_green = cur_num_green
            prev_num_lg = cur_num_lg
            list_num_white.append(cur_num_white)
            list_num_red.append(cur_num_red)
            list_num_gray.append(cur_num_gray)
            list_num_green.append(cur_num_green)
            list_num_lg.append(cur_num_lg)
            print("listnumwhite", list_num_white)
            print("listnumblack", list_num_red)
            print("listnumgray", list_num_gray)
            print("listnumgreen", list_num_green)
            print("listnumlg", list_num_lg)
            step = step + 1

        if change == 0:
            stop = 1
    return list_num_gray, n

def Sim_CM6(our_graph, num_red, k, dict_args, dict_counter_measure):
    our_graph = GetInitialOpinions(graph=our_graph, num_red=num_red, dict_args=dict_args,
                                   dict_counter_measure=dict_counter_measure)
    # set initial variables
    stop = 0
    phase = 0
    n = our_graph.number_of_nodes()
    count = 0
    cur_num_red = num_red
    cur_num_gray = 0
    cur_num_white = n - cur_num_red
    list_num_gray = [cur_num_gray]
    list_num_red = [cur_num_red]
    list_num_white = [cur_num_white]
    step = 1
    while stop != 1:
        phase = phase + 1
        change = 0
        for round in range(k + 1):
            for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                # the node has been black for k rounds and becomes gray
                if our_graph.nodes[node]['stamp'] == k:
                    our_graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                    cur_num_gray = cur_num_gray + 1
                    cur_num_red = cur_num_red - 1
                    change = change + 1
                else:
                    neighlist = list(our_graph.adj[node])
                    # only consider the white neighbors these are the only ones that can be influenced
                    for neigh in neighlist:
                        if our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE:
                            # manually add the nodes to their own neighborhoods
                            neighset = set(our_graph.adj[node])
                            neighset.add(node)
                            neighsetneigh = set(our_graph.adj[neigh])
                            neighsetneigh.add(neigh)
                            intersection_neigh = neighset.intersection(neighsetneigh)
                            union_neigh = neighset.union(neighsetneigh)
                            jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                            count = count + 1
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = float(jaccard_sim / denom)
                            rand = random.random()
                            if rand < r:
                                our_graph.nodes[neigh]['hit_counter'] = our_graph.nodes[neigh]['hit_counter'] + 1
                            else:
                                our_graph.nodes[neigh]['hit_counter'] = our_graph.nodes[neigh]['hit_counter']
            for node in our_graph.nodes:
                if our_graph.nodes[node]['hit_counter'] >= 2:
                    our_graph.nodes[node]['vote'] = NODE_COLOR_RED
                    change = change+ 1
                    cur_num_red = cur_num_red + 1
                    cur_num_white = cur_num_white - 1
            for node in our_graph.nodes:
                our_graph.nodes[node]['hit_counter'] = 0

            list_num_white.append(cur_num_white)
            list_num_red.append(cur_num_red)
            list_num_gray.append(cur_num_gray)
            print("listnumwhite", list_num_white)
            print("listnumblack", list_num_red)
            print("listnumgray", list_num_gray)
            step = step + 1
            # print("change", change)
        print("change", change)
        if change == 0:
            stop = 1
    return list_num_gray, n

def Simulation(our_graph, num_red, k, dict_args, dict_counter_measure):
    if dict_counter_measure['id'] == COUNTER_MEASURE_FACTCHECKING:
        print("counter measure factchecking")
        graylist, n = Sim_CM5(graph=our_graph, num_red=num_red, k=k,
                                                        dict_args=dict_args, dict_counter_measure=dict_counter_measure)
        return graylist, n
    if dict_counter_measure['id'] == COUNTER_MEASURE_ORANGE_NODES:
        print("counter measure orange nodes")
        graylist, n = Sim_CM1(our_graph=our_graph, num_red=num_red,k=k, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
        return graylist, n
    if dict_counter_measure['id'] == COUNTER_MEASURE_COMMUNITY_DETECTION:
        print("counter measure community detection")
        graylist, n = Sim_CM2(our_graph=our_graph, num_red=num_red, k=k, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
        return graylist, n
    if dict_counter_measure['id'] == COUNTER_MEASURE_ACCURACYFLAG:
        print('counter measure accuracy flag')
        graylist,n = Sim_CM3(our_graph=our_graph, num_red=num_red, k=k, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
        return graylist,n
    if dict_counter_measure['id'] == COUNTER_MEASURE_GREEN_NODES:
        print("counter measure green nodes")
        print("delay", dict_counter_measure['start_time'])
        graylist,n = Sim_CM4(our_graph=our_graph, num_red=num_red, k=k, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
        return graylist,n
    if dict_counter_measure['id'] == COUNTER_MEASURE_HEAR_FROM_AT_LEAST_TWO:
        graylist,n= Sim_CM6(our_graph=our_graph, num_red=num_red, k=k, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
        return graylist,n
    if dict_counter_measure['id'] == COUNTER_MEASURE_NONE:
        print("counter measure none")
        graylist,n = Sim_NOCM(our_graph=our_graph, num_red=num_red, k=k, dict_args=dict_args, dict_counter_measure=dict_counter_measure)
        return graylist,n



