# imports (START)
import networkx as nx
import networkit as nk
import os
import random
from decimal import Decimal
import powerlaw
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import math
from Utility.dataset_setup import *
from statistics import mean
import time

# imports (END)


# Constants showing the types of counter-measures (START)
COUNTER_MEASURE_NONE = 0
COUNTER_MEASURE_GREEN_INFORMATION = 1
COUNTER_MEASURE_HEAR_FROM_AT_LEAST_TWO = 2
COUNTER_MEASURE_DELAYED_SPREADING = 3
COUNTER_MEASURE_COMMUNITY_DETECTION = 4
COUNTER_MEASURE_DOUBT_SPREADING = 5
# Constants showing the types of counter-measures (END)
# Constants showing the color of nodes (START)
NODE_COLOR_RED = 1
NODE_COLOR_WHITE = -1
NODE_COLOR_ORANGE = 0
NODE_COLOR_RESERVED = 2
NODE_COLOR_GREEN = 3


# Constants showing the color of nodes (END)


# node colour IDs
def ringOfCliques(num_cliques, clique_size):
    """
    This function returns a ring-of-cliques graph where:
    num_cliques: the number of cliques
    clique_size: the number of nodes in each clique
    """
    G = nx.ring_of_cliques(num_cliques, clique_size)
    return G


# ???
def KCliqueExpander(j, c, d):
    KC = ringOfCliques(j, c)
    # print("KCLIQUE", KC, KC.nodes())
    # print(j*c)
    mapping = dict(zip(KC, range(0, KC.number_of_nodes() - 1)))
    # temp_graph = nx.relabel_nodes(temp_graph, mapping)
    RRG = nx.random_regular_graph(d, j * c)
    RRG = nx.relabel_nodes(RRG, mapping)
    # print("RRG", RRG, RRG.nodes())
    G = nx.compose(RRG, KC)
    return G


def GetSNlikeGraph(realworld_graph, type_graph):
    # Generating the graph
    temp_graph = nx.read_edgelist(realworld_graph["loc"], create_using=nx.Graph(), nodetype=int)
    if (realworld_graph["directed"]):
        temp_graph = temp_graph.to_undirected()  # Make directed graphs undirected
    mapping = dict(zip(temp_graph, range(0, temp_graph.number_of_nodes())))
    temp_graph = nx.relabel_nodes(temp_graph, mapping)
    n = temp_graph.number_of_nodes()
    total = sum(j for i, j in list(temp_graph.degree(temp_graph.nodes)))
    av_deg = total / temp_graph.number_of_nodes()
    # print("av_deg", av_deg)
    p = total / (temp_graph.number_of_nodes() * (temp_graph.number_of_nodes() - 1))
    d = round(n * p / 2) * 2
    degrees = {}
    for node in temp_graph.nodes():
        key = len(temp_graph.adj[node])
        degrees[key] = degrees.get(key, 0) + 1

    max_degree = max(degrees.keys(), key=int)
    min_degree = min(degrees.keys(), key=int)
    num_nodes = []
    for i in range(1, max_degree + 1):
        num_nodes.append(degrees.get(i, 0))

    fit = powerlaw.Fit(num_nodes)
    # print(fit.power_law.alpha)

    if type_graph == TYPE_ERDOS_RENYI:
        our_graph = nx.fast_gnp_random_graph(n=temp_graph.number_of_nodes(), p=p)
    elif type_graph == TYPE_BA:
        our_graph = nx.barabasi_albert_graph(n=temp_graph.number_of_nodes(), m=int(av_deg))
    elif type_graph == TYPE_LFR:
        dict_args = {"n": temp_graph.number_of_nodes(), "tau1": fit.power_law.alpha, "tau2": fit.power_law.alpha,
                     "mu": 0.4, "average_degree": av_deg, "min_degree": min_degree,
                     "min_community": math.sqrt(temp_graph.number_of_nodes()),
                     "tol": 0.04, "max_iters": 1000, "seed": None}

        our_graph = nx.LFR_benchmark_graph(n=dict_args["n"], tau1=dict_args["tau1"], tau2=dict_args["tau2"],
                                           mu=dict_args["mu"], average_degree=dict_args["average_degree"],
                                           min_community=dict_args["min_community"],
                                           tol=dict_args["tol"], max_iters=dict_args["max_iters"],
                                           seed=dict_args["seed"])

        print("nodes: " + str(len(our_graph.nodes())))
        print("edges: " + str(our_graph.number_of_edges()))


    elif type_graph == TYPE_SOCIAL_NETWORK:
        our_graph = temp_graph
    elif type_graph == TYPE_D_REGULAR_RANDOM_GRAPH:
        our_graph = nx.random_regular_graph(d=d, n=n)
    elif type_graph == TYPE_HRG:
        hg = nk.generators.HyperbolicGenerator(n=temp_graph.number_of_nodes(), k=av_deg, gamma=2.5, T=0.6)
        hgG = hg.generate()
        our_graph = nk.nxadapter.nk2nx(hgG)
    elif type_graph == TYPE_CYCLIC:
        our_graph = nx.cycle_graph(n)
    elif type_graph == TYPE_RING_OF_CLIQUES:
        our_graph = ringOfCliques(math.sqrt(n), math.sqrt(n))  # using just a rule of thumb and creating sqrt(n) cliques
        # with sqrt(n) nodes in them.
    # ???
    # elif type_graph == "KCliqueExpander":
    #     our_graph = KCliqueExpander(dict_args["num_cliques"], dict_args["clique_size"], d)
    # ???
    elif type_graph == TYPE_COMPLETE:
        our_graph = nx.complete_graph(n)
    elif type_graph == TYPE_MODERATELY_EXPANDER:
        our_graph = moderatelyExpander(degree_of_each_supernode=math.log(n),
                                       number_of_supernodes=math.sqrt(n),
                                       nodes_in_clique=math.sqrt(n))  # using just a rule of thumb and creating sqrt(n)
        # cliques with the degrees of log(n) and sqrt(n) nodes withing them.
    return our_graph


def getGraph(dict_args):
    # n = temp_graph.number_of_nodes()
    # mapping = dict(zip(temp_graph, range(0, temp_graph.number_of_nodes())))
    # temp_graph = nx.relabel_nodes(temp_graph, mapping)

    # total = sum(j for i, j in list(temp_graph.degree(temp_graph.nodes)))
    # av_deg = total / temp_graph.number_of_nodes()
    # print("av_deg", av_deg)
    # p = total / (temp_graph.number_of_nodes() * (temp_graph.number_of_nodes() - 1))
    # d = round(n * p / 2) * 2

    # degrees = {}
    # for node in temp_graph.nodes():
    #     key = len(temp_graph.adj[node])
    #     degrees[key] = degrees.get(key, 0) + 1
    #
    # max_degree = max(degrees.keys(), key=int)
    # min_degree = min(degrees.keys(), key=int)
    # num_nodes = []
    # for i in range(1, max_degree + 1):
    #     num_nodes.append(degrees.get(i, 0))

    if dict_args["type"] == TYPE_ERDOS_RENYI:
        our_graph = nx.fast_gnp_random_graph(n=dict_args["n"], p=dict_args["p"])
    elif dict_args["type"] == TYPE_BA:
        our_graph = nx.barabasi_albert_graph(n=dict_args["n"], m=dict_args["m"])
    elif dict_args["type"] == TYPE_D_REGULAR_RANDOM_GRAPH:
        our_graph = nx.random_regular_graph(d=dict_args["d"], n=dict_args["n"])
    elif dict_args["type"] == TYPE_HRG:
        hg = nk.generators.HyperbolicGenerator(n=dict_args["n"], k=dict_args["avg_degree"], gamma=2.5, T=0.6)
        hgG = hg.generate()
        our_graph = nk.nxadapter.nk2nx(hgG)
    elif dict_args["type"] == TYPE_SOCIAL_NETWORK:
        our_graph = nx.read_edgelist(dict_args["loc"], create_using=nx.Graph(), nodetype=int)
    elif dict_args["type"] == TYPE_CYCLIC:
        our_graph = nx.cycle_graph(dict_args["n"])
    elif dict_args["type"] == TYPE_RING_OF_CLIQUES:
        our_graph = ringOfCliques(dict_args["num_cliques"], dict_args["clique_size"])
    # elif type_graph == "KCliqueExpander":
    #     our_graph = KCliqueExpander(dict_args["num_cliques"], dict_args["clique_size"], d)
    elif dict_args["type"] == TYPE_COMPLETE:
        our_graph = nx.complete_graph(dict_args["n"])
    elif dict_args["type"] == TYPE_MODERATELY_EXPANDER:
        our_graph = moderatelyExpander(degree_of_each_supernode=dict_args["degree_of_supernodes"],
                                       number_of_supernodes=dict_args["number_of_supernodes"],
                                       nodes_in_clique=dict_args["nodes_in_clique"])
    elif dict_args["type"] == TYPE_LFR:
        our_graph = nx.LFR_benchmark_graph(n=dict_args["n"], tau1=dict_args["tau1"], tau2=dict_args["tau2"],
                                        mu=dict_args["mu"], average_degree=dict_args["average_degree"],
                                        min_community=dict_args["min_community"],
                                        tol=dict_args["tol"], max_iters=dict_args["max_iters"],
                                        seed=dict_args["seed"])
    # print("edges: " + str(our_graph.number_of_edges()))
    # print("nodes: " + str(len(our_graph.nodes())))
    # print("is_connected: " + str(nx.is_connected(our_graph)))

    # print("average degree: " + str((mean([val for (node, val) in our_graph.degree()]))))
    # plt.savefig("filenameLFR.png")

    return our_graph


# Just a backup of out previous function
def GetGraph_backup(graph, type_graph, dict_args):
    temp_graph = nx.read_edgelist(os.path.join(ABS_PATH, graph), create_using=nx.Graph(), nodetype=int)
    n = temp_graph.number_of_nodes()
    mapping = dict(zip(temp_graph, range(0, temp_graph.number_of_nodes())))
    temp_graph = nx.relabel_nodes(temp_graph, mapping)

    total = sum(j for i, j in list(temp_graph.degree(temp_graph.nodes)))
    av_deg = total / temp_graph.number_of_nodes()
    # print("av_deg", av_deg)
    p = total / (temp_graph.number_of_nodes() * (temp_graph.number_of_nodes() - 1))
    d = round(n * p / 2) * 2

    degrees = {}
    for node in temp_graph.nodes():
        key = len(temp_graph.adj[node])
        degrees[key] = degrees.get(key, 0) + 1

    max_degree = max(degrees.keys(), key=int)
    min_degree = min(degrees.keys(), key=int)
    num_nodes = []
    for i in range(1, max_degree + 1):
        num_nodes.append(degrees.get(i, 0))

    # fit = powerlaw.Fit(num_nodes)
    # print(fit.power_law.alpha)

    if type_graph == 'ER':
        # print("ER graph")
        our_graph = nx.fast_gnp_random_graph(n=temp_graph.number_of_nodes(), p=p)
    elif type_graph == 'BA':
        # print("BA graph")
        our_graph = nx.barabasi_albert_graph(n=temp_graph.number_of_nodes(), m=int(av_deg))
    elif type_graph == "DREG":
        our_graph = nx.random_regular_graph(d=d, n=n)
    elif type_graph == "HRG":
        hg = nk.generators.HyperbolicGenerator(n=temp_graph.number_of_nodes(), k=av_deg, gamma=2.5, T=0.6)
        hgG = hg.generate()
        our_graph = nk.nxadapter.nk2nx(hgG)


    elif type_graph == 'SN':
        our_graph = temp_graph




    elif type_graph == "cycle":
        our_graph = nx.cycle_graph(n)
    elif type_graph == "KClique":
        our_graph = ringOfCliques(dict_args["num_cliques"], dict_args["clique_size"])
    elif type_graph == "KCliqueExpander":
        our_graph = KCliqueExpander(dict_args["num_cliques"], dict_args["clique_size"], d)
    elif type_graph == "Complete":
        our_graph = nx.complete_graph(dict_args["n"])
    elif type_graph == "moderatelyExpander":
        our_graph = moderatelyExpander(degree_of_each_supernode=dict_args["degree_of_supernodes"],
                                       number_of_supernodes=dict_args["number_of_supernodes"],
                                       nodes_in_clique=dict_args["nodes_in_clique"])
    # elif type_graph == "LFR":
    # dict_args:
    #   n:  int Number of nodes in the created graph.
    #   tau1:   float   Power law exponent for the degree distribution of the created graph.
    #   This value must be strictly greater than one.
    #   tau2:   float   Power law exponent for the community size distribution in the created graph.
    #   This value must be strictly greater than one.
    #   mu: float   Fraction of inter-community edges incident to each node.
    #   This value must be in the interval [0, 1].
    #   average_degree: float   Desired average degree of nodes in the created graph.
    #   This value must be in the interval [0, n].
    #   min_degree: int Minimum degree of nodes in the created graph. This value must be in the interval [0, n].
    #   Exactly one of this and average_degree must be specified, otherwise a NetworkXError is raised.
    #   max_degree: int Maximum degree of nodes in the created graph.
    #   min_community:  int Minimum size of communities in the graph.
    #   max_community:  int Maximum size of communities in the graph.
    #   tol:    float   Tolerance when comparing floats, specifically when comparing average degree values.
    #   max_iters:  int Maximum number of iterations to try to create the community sizes, degree distribution,
    #   and community affiliations.
    #   seed:   integer, random_state, or None (default)    Indicator of random number generation state.
    # our_graph = nx.LFR_benchmark_graph(n=dict_args["n"], tau1=dict_args["tau1"], tau2=dict_args["tau2"],
    #                                    mu=dict_args["mu"], average_degree=dict_args["average_degree"],
    #                                    min_community=dict_args["min_community"],
    #                                    tol=dict_args["tol"], max_iters=dict_args["max_iters"],
    #                                    seed=dict_args["seed"])
    # print("edges: " + str(our_graph.number_of_edges()))
    # print("nodes: " + str(len(our_graph.nodes())))
    # print("is_connected: " + str(nx.is_connected(our_graph)))

    # print("average degree: " + str((mean([val for (node, val) in our_graph.degree()]))))
    # plt.savefig("filenameLFR.png")

    return our_graph

# I stopped here
def GetInitialOpinions(graph, num_red, gray_p):
    num_red_c = 0
    for node in graph.nodes:
        graph.nodes[node]['hit_counter'] = 0
        graph.nodes[node]['sleep_timer'] = 0
        if (random.random() < gray_p):
            # these are the nodes that will have vote gray and thus are not affected by the neighbours
            graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
        else:
            # the nodes that will have vote black or white
            graph.nodes[node]['vote'] = NODE_COLOR_WHITE
    while (num_red_c < num_red):
        r_node = random.randint(0, graph.number_of_nodes() - 1)
        if graph.nodes[r_node]['vote'] == NODE_COLOR_WHITE:
            graph.nodes[r_node]['vote'] = NODE_COLOR_RED
            # print("black node's degree: ", graph.degree(r_node))
            num_red_c += 1

    return graph


def moderatelyExpander(degree_of_each_supernode, number_of_supernodes, nodes_in_clique):
    print("degree of each supernode", degree_of_each_supernode)
    H = nx.random_regular_graph(d=degree_of_each_supernode, n=number_of_supernodes)

    G = nx.complete_graph(n=nodes_in_clique)
    H_nodes = list(H.nodes())

    # print("nodes : " + str(H_nodes))
    for i in range(len(H_nodes) - 1):
        G = nx.disjoint_union(G, nx.complete_graph(n=nodes_in_clique))
    for i in H_nodes:
        edges_i = list(H.edges(i))
        # print("edges in " + str(i))
        # print(edges_i)
        for j in range(len(edges_i)):
            # print(str(edges_i[j]) + " => (" + str(edges_i[j][0] * nodes_in_clique) + ", " + str(
            #    edges_i[j][1] * nodes_in_clique) + ")")
            G.add_edge(
                random.randint(edges_i[j][0] * nodes_in_clique, edges_i[j][0] * nodes_in_clique + nodes_in_clique - 1),
                random.randint(edges_i[j][1] * nodes_in_clique, edges_i[j][1] * nodes_in_clique + nodes_in_clique - 1))
        H.remove_node(i)

    # nx.draw(G)
    # plt.savefig("filename.png")
    return G


def Simulation_no_countermeasure(graph, type_graph, num_red, k, dict_args):
    our_graph = getGraph(graph=graph, type_graph=type_graph, dict_args=dict_args)
    our_graph = GetInitialOpinions(graph=our_graph, num_red=num_red, gray_p=0)
    stop = 0
    phase = 0
    sum_jaccard_sim = 0
    count = 0
    n = our_graph.number_of_nodes()

    for node in our_graph.nodes:
        # stamp indicates the number of rounds a node has been black
        our_graph.nodes[node]['stamp'] = 0

    cur_num_red = num_red
    cur_num_gray = 0
    cur_num_white = our_graph.number_of_nodes() - num_red
    list_num_gray = [cur_num_gray]
    list_num_black = [cur_num_red]
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
                    our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                    cur_num_gray = cur_num_gray + 1
                    cur_num_red = cur_num_red - 1
                else:
                    neighlist = list(our_graph.adj[node])
                    # only consider the white neighbors these are the only ones that can be influenced
                    for neigh in [neigh for neigh in neighlist if
                                  our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE]:
                        # manually add the nodes to their own neighborhoods
                        neighset = set(our_graph.adj[node])
                        neighset.add(node)
                        neighsetneigh = set(our_graph.adj[neigh])
                        neighsetneigh.add(neigh)
                        intersection_neigh = neighset.intersection(neighsetneigh)
                        union_neigh = neighset.union(neighsetneigh)
                        jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                        sum_jaccard_sim += jaccard_sim
                        count = count + 1
                        denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                        r = (jaccard_sim / denom)
                        rand = random.random()
                        if rand < r:
                            our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                            change = change + 1
                            cur_num_red = cur_num_red + 1
                            cur_num_white = cur_num_white - 1

            list_num_white.append(cur_num_white)
            list_num_black.append(cur_num_red)
            list_num_gray.append(cur_num_gray)

            step += 1
            if change == 0:
                stop = 1

    return list_num_gray, n


def simulation(realworld_graph=None, num_red=1, num_orange=0, k=5, dict_args={"type": TYPE_ERDOS_RENYI, "n": 200},
               dict_counter_measure={"id": COUNTER_MEASURE_NONE}, seed=None):
    """
    This function is the implementation of the proposed model and counter measures. The followings describe each
     parameter's use:
    realworld_graph: we use a dictionary similar to the one defined in `dataset_setup.py` to import external datasets.
                    This parameter should be equal to None in case the experiment is intended to run just on synthetic
                    networks. If both this parameter and `dict_args` are set, the simulation will run on a synthetic
                    network with characteristics similar to the given realworld_graph.
    num_red: number of red colored nodes in the first epoch of the simulation.
    num_orange: number of orange colored nodes in the first epoch of the simulation.
    k: the number of epoches that takes until a red node turns orange.
    dict_args: a dictionary of the parameters to be used for creating a synthetic network (explained in `main.py`).
                If both this parameter and `realworld_graph` are set, the simulation will run on a synthetic network
                with characteristics similar to the given realworld_graph.
    dict_counter_measure: a dictionary of the parameters to be used for running the counter measure.
    seed: The seed value for creating pseudo-random numbers in the process.
    """
    # setting random seed value (START)
    random.seed(seed)
    # setting random seed value (END)

    # Generating the graph (START)
    if (realworld_graph != None and dict_args != None):
        pass
    # create synthetic graphs similar to the real world graph
    elif (realworld_graph == None and dict_args != None):
        pass
    elif (realworld_graph != None and dict_args == None):
        pass
    our_graph = getGraph(dict_args=dict_args)
    # Generating the graph (START)

    # generate the initial opinions of the graph
    our_graph = GetInitialOpinions(graph=our_graph, num_red=num_red, gray_p=dict_counter_measure['gray_p'])

    # now we consider the update rule
    stop = 0
    phase = 0
    sum_jaccard_sim = 0
    count = 0

    for node in our_graph.nodes:
        # stamp indicates the number of rounds a node has been black
        our_graph.nodes[node]['stamp'] = 0
        if (dict_counter_measure["id"] == COUNTER_MEASURE_DELAYED_SPREADING):
            if (our_graph.nodes[node]['vote'] == NODE_COLOR_RED):
                our_graph.nodes[node]['sleep_timer'] = dict_counter_measure["sleep_timer"]

    rednodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]
    cur_num_red = len(rednodes_initial)

    gray_nodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_ORANGE]
    cur_num_gray = len(gray_nodes_initial)

    whitenodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_WHITE]
    cur_num_white = len(whitenodes_initial)

    # green_nodes_initial = []
    cur_num_green = 0
    # list_num_green = []
    if (dict_counter_measure["id"] == COUNTER_MEASURE_GREEN_INFORMATION):
        while (cur_num_green < dict_counter_measure["num_green"]):
            r = random.randint(0, our_graph.number_of_nodes() - 1)
            # turn a white node to a green node
            if (our_graph.nodes[r]['vote'] == NODE_COLOR_WHITE):
                our_graph.nodes[r]['vote'] = NODE_COLOR_GREEN
                green_nodes_initial = [r]
                cur_num_green += 1
                cur_num_white -= 1
    threshold_detection = 0
    threshold_block = 0
    cs = []
    if (dict_counter_measure["id"] == COUNTER_MEASURE_COMMUNITY_DETECTION):
        threshold_detection = dict_counter_measure["threshold_detection"]
        threshold_block = dict_counter_measure["threshold_block"]
        cs = nx_comm.louvain_communities(our_graph)
        c_ind = 0
        for c in cs:
            for n in c:
                our_graph.nodes[n]["comm"] = c_ind
            c_ind += 1

    doubt_counter = 0
    negative_doubt_shift = 0.0
    positive_doubt_shift = 0.0
    doubt_spreader = []
    doubt_ls = []
    if (dict_counter_measure["id"] == COUNTER_MEASURE_DOUBT_SPREADING):
        negative_doubt_shift = dict_counter_measure["negative_doubt_shift"]
        positive_doubt_shift = dict_counter_measure["positive_doubt_shift"]
        for node in our_graph.nodes():
            our_graph.nodes[node]["doubt"] = random.normalvariate(0.5, 0.16)
            our_graph.nodes[node]['origin'] = []
            if (our_graph.nodes[node]["doubt"] > 1):
                our_graph.nodes[node]["doubt"] = 1
            if (our_graph.nodes[node]["doubt"] < 0):
                our_graph.nodes[node]["doubt"] = 0
            doubt_ls.append(our_graph.nodes[node]["doubt"])
        plt.clf()
        plt.hist(doubt_ls, bins=30)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    list_num_gray = [cur_num_gray]
    list_num_black = [cur_num_red]
    list_num_white = [cur_num_white]
    list_num_green = [cur_num_green]
    # now (in contrast to the prev project) we proceed in phases, where each phase consists of k rounds.
    step = 1
    while stop != 1:
        phase = phase + 1
        # print("phase", phase)
        change = 0
        for round in range(k + 1):
            # print("round", round)
            if (dict_counter_measure["id"] == COUNTER_MEASURE_NONE):
                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been black for k rounds and becomes gray
                    if our_graph.nodes[node]['stamp'] == k:
                        # print("becoming gray", node)
                        our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                        cur_num_gray = cur_num_gray + 1
                        cur_num_red = cur_num_red - 1
                    else:
                        # print("node", node)
                        neighlist = list(our_graph.adj[node])
                        # print("neighlist",neighlist)
                        # only consider the white neighbors these are the only ones that can be influenced
                        for neigh in [neigh for neigh in neighlist if
                                      our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE]:
                            # manually add the nodes to their own neighborhoods
                            neighset = set(our_graph.adj[node])
                            neighset.add(node)
                            neighsetneigh = set(our_graph.adj[neigh])
                            neighsetneigh.add(neigh)
                            intersection_neigh = neighset.intersection(neighsetneigh)
                            union_neigh = neighset.union(neighsetneigh)
                            jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                            sum_jaccard_sim += jaccard_sim
                            count = count + 1
                            # print("jaccard_sim", jaccard_sim)
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = (jaccard_sim / denom)
                            rand = random.random()
                            # print("r and rand", r, rand)
                            if rand < r:
                                # print("neigh",neigh) #see if there are duplicates
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                change = change + 1
                                cur_num_red = cur_num_red + 1
                                cur_num_white = cur_num_white - 1
                                # print("change", change)
            elif (dict_counter_measure["id"] == COUNTER_MEASURE_GREEN_INFORMATION):
                for node in [node for node in our_graph.nodes if
                             our_graph.nodes[node]['vote'] == NODE_COLOR_RED or our_graph.nodes[node][
                                 'vote'] == NODE_COLOR_GREEN]:
                    if (our_graph.nodes[node]["vote"] == NODE_COLOR_GREEN):
                        if (step >= dict_counter_measure["start_time"]):
                            our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                            # **
                            if our_graph.nodes[node]['stamp'] == k:
                                our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                                cur_num_gray = cur_num_gray + 1
                                cur_num_green = cur_num_green - 1
                            else:
                                # print("node", node)
                                neighlist = list(our_graph.adj[node])
                                # only consider the white neighbors these are the only ones that can be influenced
                                for neigh in neighlist:
                                    if (our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE):
                                        # manually add the nodes to their own neighborhoods
                                        neighset = set(our_graph.adj[node])
                                        neighset.add(node)
                                        neighsetneigh = set(our_graph.adj[neigh])
                                        neighsetneigh.add(neigh)
                                        intersection_neigh = neighset.intersection(neighsetneigh)
                                        union_neigh = neighset.union(neighsetneigh)
                                        jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                                        sum_jaccard_sim += jaccard_sim
                                        count = count + 1
                                        # print("jaccard_sim", jaccard_sim)
                                        denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                                        r = (jaccard_sim / denom)
                                        rand = random.random()
                                        # print("r and rand", r, rand)
                                        if rand < r:
                                            # print("neigh",neigh) #see if there are duplicates
                                            our_graph.nodes[neigh]['vote'] = NODE_COLOR_GREEN
                                            change = change + 1
                                            cur_num_green = cur_num_green + 1
                                            cur_num_white = cur_num_white - 1
                                            # print("change", change)
                    elif (our_graph.nodes[node]["vote"] == NODE_COLOR_RED):
                        our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                        # the node has been black for k rounds and becomes gray
                        if our_graph.nodes[node]['stamp'] == k:
                            # print("becoming gray", node)
                            our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                            cur_num_gray = cur_num_gray + 1
                            cur_num_red = cur_num_red - 1
                        else:
                            # print("node", node)
                            neighlist = list(our_graph.adj[node])
                            # print("neighlist",neighlist)
                            # only consider the white neighbors these are the only ones that can be influenced
                            for neigh in neighlist:
                                if (our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE):
                                    # manually add the nodes to their own neighborhoods
                                    neighset = set(our_graph.adj[node])
                                    neighset.add(node)
                                    neighsetneigh = set(our_graph.adj[neigh])
                                    neighsetneigh.add(neigh)
                                    intersection_neigh = neighset.intersection(neighsetneigh)
                                    union_neigh = neighset.union(neighsetneigh)
                                    jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                                    sum_jaccard_sim += jaccard_sim
                                    count = count + 1
                                    # print("jaccard_sim", jaccard_sim)
                                    denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                                    r = (jaccard_sim / denom)
                                    rand = random.random()
                                    # print("r and rand", r, rand)
                                    if rand < r:
                                        # print("neigh",neigh) #see if there are duplicates
                                        our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                        change = change + 1
                                        cur_num_RED = cur_num_red + 1
                                        cur_num_white = cur_num_white - 1
                                        # print("change", change)
            elif (dict_counter_measure["id"] == COUNTER_MEASURE_HEAR_FROM_AT_LEAST_TWO):
                hit_nodes = []
                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been black for k rounds and becomes gray
                    if our_graph.nodes[node]['stamp'] == k:
                        # print("becoming gray", node)
                        our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                        cur_num_gray = cur_num_gray + 1
                        cur_num_RED = cur_num_red - 1
                    else:
                        # print("node", node)
                        neighlist = list(our_graph.adj[node])
                        # print("neighlist",neighlist)
                        # only consider the white neighbors these are the only ones that can be influenced
                        for neigh in [neigh for neigh in neighlist if
                                      our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE]:
                            # manually add the nodes to their own neighborhoods
                            neighset = set(our_graph.adj[node])
                            neighset.add(node)
                            neighsetneigh = set(our_graph.adj[neigh])
                            neighsetneigh.add(neigh)
                            intersection_neigh = neighset.intersection(neighsetneigh)
                            union_neigh = neighset.union(neighsetneigh)
                            jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                            sum_jaccard_sim += jaccard_sim
                            count = count + 1
                            # print("jaccard_sim", jaccard_sim)
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = (jaccard_sim / denom)
                            rand = random.random()
                            # print("r and rand", r, rand)
                            if rand < r:
                                our_graph.nodes[neigh]['hit_counter'] += 1
                                hit_nodes.append(neigh)

                for hit_node in hit_nodes:
                    if (our_graph.nodes[hit_node]['hit_counter'] >= 2):
                        our_graph.nodes[hit_node]['vote'] = NODE_COLOR_RED
                        change = change + 1
                        cur_num_red = cur_num_red + 1
                        cur_num_white = cur_num_white - 1
                    our_graph.nodes[hit_node]['hit_counter'] = 0

                    # print("change", change)
            elif (dict_counter_measure["id"] == COUNTER_MEASURE_DELAYED_SPREADING):
                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                    if our_graph.nodes[node]['sleep_timer'] != 0:
                        our_graph.nodes[node]['sleep_timer'] -= 1
                        continue
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been black for k rounds and becomes gray
                    if our_graph.nodes[node]['stamp'] == k:
                        # print("becoming gray", node)
                        our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                        cur_num_gray = cur_num_gray + 1
                        cur_num_red = cur_num_red - 1
                    else:
                        # print("node", node)
                        neighlist = list(our_graph.adj[node])
                        # print("neighlist",neighlist)
                        # only consider the white neighbors these are the only ones that can be influenced
                        for neigh in [neigh for neigh in neighlist if
                                      our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE]:
                            # manually add the nodes to their own neighborhoods
                            neighset = set(our_graph.adj[node])
                            neighset.add(node)
                            neighsetneigh = set(our_graph.adj[neigh])
                            neighsetneigh.add(neigh)
                            intersection_neigh = neighset.intersection(neighsetneigh)
                            union_neigh = neighset.union(neighsetneigh)
                            jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                            sum_jaccard_sim += jaccard_sim
                            count = count + 1
                            # print("jaccard_sim", jaccard_sim)
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = (jaccard_sim / denom)
                            rand = random.random()
                            # print("r and rand", r, rand)
                            if rand < r:
                                # print("neigh",neigh) #see if there are duplicates
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                our_graph.nodes[neigh]['sleep_timer'] = dict_counter_measure["sleep_timer"]
                                change = change + 1
                                cur_num_red = cur_num_red + 1
                                cur_num_white = cur_num_white - 1
            elif (dict_counter_measure["id"] == COUNTER_MEASURE_COMMUNITY_DETECTION):
                black_nodes = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]
                maxes = []
                if (len(black_nodes) >= (int)(our_graph.number_of_nodes() * threshold_detection)):
                    black_ratio_per_community = []
                    counter = 0
                    for c in cs:
                        black_ratio_per_community.append(0)
                        for n in c:
                            if (our_graph.nodes[n]['vote'] == NODE_COLOR_RED):
                                black_ratio_per_community[counter] += 1
                        black_ratio_per_community[counter] = black_ratio_per_community[counter] / len(c)

                        if black_ratio_per_community[counter] >= threshold_block:
                            maxes.append(counter)
                        counter += 1
                for node in black_nodes:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been black for k rounds and becomes gray
                    if our_graph.nodes[node]['stamp'] == k:
                        # print("becoming gray", node)
                        our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                        cur_num_gray = cur_num_gray + 1
                        cur_num_red = cur_num_red - 1
                    else:
                        # print("node", node)
                        neighlist = list(our_graph.adj[node])
                        # print("neighlist",neighlist)
                        # only consider the white neighbors these are the only ones that can be influenced
                        for neigh in [neigh for neigh in neighlist if
                                      our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE]:
                            # manually add the nodes to their own neighborhoods
                            neighset = set(our_graph.adj[node])
                            neighset.add(node)
                            neighsetneigh = set(our_graph.adj[neigh])
                            neighsetneigh.add(neigh)
                            intersection_neigh = neighset.intersection(neighsetneigh)
                            union_neigh = neighset.union(neighsetneigh)
                            jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                            sum_jaccard_sim += jaccard_sim
                            count = count + 1
                            # print("jaccard_sim", jaccard_sim)
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = (jaccard_sim / denom)
                            rand = random.random()
                            # print("r and rand", r, rand)
                            if rand < r:
                                if (our_graph.nodes[node]["comm"] in maxes):
                                    if (our_graph.nodes[neigh]["comm"] != our_graph.nodes[node]["comm"]):
                                        print("the spread of rumor from #" + str(node) + " to #" +
                                              str(neigh) + " has been blocked because #" + str(
                                            node) + " is in the blocked community")
                                        continue
                                # print("neigh",neigh) #see if there are duplicates
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                change = change + 1
                                cur_num_red = cur_num_red + 1
                                cur_num_white = cur_num_white - 1
                                # print("change", change)
            elif (dict_counter_measure["id"] == COUNTER_MEASURE_DOUBT_SPREADING):
                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been black for k rounds and becomes gray
                    if our_graph.nodes[node]['stamp'] == k:
                        # print("becoming gray", node)
                        our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                        cur_num_gray = cur_num_gray + 1
                        cur_num_red = cur_num_red - 1
                    else:
                        # print("node", node)
                        neighlist = list(our_graph.adj[node])
                        # print("neighlist",neighlist)
                        # only consider the white neighbors these are the only ones that can be influenced
                        for neigh in [neigh for neigh in neighlist if
                                      our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE]:
                            # manually add the nodes to their own neighborhoods
                            neighset = set(our_graph.adj[node])
                            neighset.add(node)
                            neighsetneigh = set(our_graph.adj[neigh])
                            neighsetneigh.add(neigh)
                            intersection_neigh = neighset.intersection(neighsetneigh)
                            union_neigh = neighset.union(neighsetneigh)
                            jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                            sum_jaccard_sim += jaccard_sim
                            count = count + 1
                            # print("jaccard_sim", jaccard_sim)
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = (jaccard_sim / denom)
                            rand = random.random()
                            # print("r and rand", r, rand)
                            if rand < r:
                                # print("neigh",neigh) #see if there are duplicates
                                rand2_doubt = random.random()
                                if (our_graph.nodes[neigh]['doubt'] < rand2_doubt):
                                    our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                    change = change + 1
                                    cur_num_red = cur_num_red + 1
                                    cur_num_white = cur_num_white - 1

                                    # reduce doubt
                                    delta_doubt = random.uniform(0, negative_doubt_shift)
                                    # print("negative delta_doubt: "+str(delta_doubt))
                                    our_graph.nodes[neigh]['doubt'] += delta_doubt
                                    our_graph.nodes[neigh]['origin'].append((doubt_counter, delta_doubt))
                                    doubt_counter += 1
                                    doubt_spreader.append(neigh)
                                    if (our_graph.nodes[neigh]['doubt'] < 0):
                                        our_graph.nodes[neigh]['doubt'] = 0
                                else:
                                    # add doubt
                                    delta_doubt = random.uniform(0, positive_doubt_shift)
                                    # print("positive delta_doubt: "+str(delta_doubt))
                                    our_graph.nodes[neigh]['doubt'] += delta_doubt
                                    our_graph.nodes[neigh]['origin'].append((doubt_counter, delta_doubt))
                                    doubt_counter += 1
                                    doubt_spreader.append(neigh)
                                    if (our_graph.nodes[neigh]['doubt'] > 1):
                                        our_graph.nodes[neigh]['doubt'] = 1

                # spread doubts
                print("Updating doubt values")
                while (len(doubt_spreader) > 0):
                    d_node = doubt_spreader.pop()
                    if our_graph.nodes[d_node]["vote"] == NODE_COLOR_ORANGE:
                        continue
                    (d, delta) = our_graph.nodes[d_node]['origin'][-1]
                    for neigh in our_graph.neighbors(d_node):
                        if (d in [x for (x, y) in our_graph.nodes[d_node]['origin']]):
                            continue
                        else:
                            delta_doubt = random.uniform(0, delta)
                            our_graph.nodes[neigh]['doubt'] += delta_doubt
                            our_graph.nodes[neigh]['origin'].append((d, delta_doubt))
                            doubt_spreader.append(neigh)
                doubt_ls = []
                for node in our_graph.nodes():
                    doubt_ls.append(our_graph.nodes[node]["doubt"])
                plt.clf()
                plt.hist(doubt_ls, bins=30)
                plt.show(block=False)
                plt.pause(1)
                plt.close()
            print("round", round, "phase", phase, cur_num_gray, cur_num_white, cur_num_red, cur_num_green)
            list_num_white.append(cur_num_white)
            list_num_black.append(cur_num_red)
            list_num_gray.append(cur_num_gray)
            list_num_green.append(cur_num_green)
            print("listnumwhite", list_num_white)
            print("listnumblack", list_num_black)
            print("listnumgray", list_num_gray)
            print("list_num_green", list_num_green)
            step += 1
        print("change", change)
        if change == 0:
            stop = 1

    # print("av jaccard", sum_jaccard_sim/count)
    return [list_num_white, list_num_black, list_num_gray, list_num_green]
