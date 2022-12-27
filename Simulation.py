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
import igraph as ig
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
NODE_COLOR_PALE_GREEN = 2
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


def getSNLikeGraph(realworld_graph, type_graph):
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
        if (dict_args["directed"]):
            our_graph = our_graph.to_undirected()  # Make directed graphs undirected
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


def getInitialOpinions(graph, num_red, orange_p):
    """
    This function receives the graph object, number of initial red nodes and the initial ratio of orange nodes and
    colors the nodes based on the input values
    graph: networkx graph
    num_red: int, number of red nodes
    orange_p: initial ratio of orange nodes
    """
    white_nodes = list()
    for node in graph.nodes:
        graph.nodes[node]['hit_counter'] = 0  # to be used in hear_from_at_least_two_neighbors counter measure
        graph.nodes[node]['sleep_timer'] = 0  # to be used in delayed_spreading counter measure
        graph.nodes[node]['stamp'] = 0  # to indicate the number of rounds a node has been red
        graph.nodes[node]['traingle'] = 0  # ????
        # Assigning orange color to a predefined percentage of nodes (START)
        if random.random() < orange_p:
            graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
        else:
            graph.nodes[node]['vote'] = NODE_COLOR_WHITE
            white_nodes.append(node)
        # Assigning orange color to a predefined percentage of nodes (END)
    # selecting `num_red` white nodes randomly and turning them red (START)
    red_nodes = random.choices(white_nodes, k=num_red)
    for red_node in red_nodes:
        graph.nodes[red_node]['vote'] = NODE_COLOR_RED
    # selecting `num_red` white nodes randomly and turning them red (END)
    # set a blocked flag to all edges to be able to track them (START)
    for edge in graph.edges:
        graph.edges[edge]["blocked"] = 0
    # set a blocked flag to all edges to be able to track them (END)
    return graph


def moderatelyExpander(degree_of_each_supernode, number_of_supernodes, nodes_in_clique):
    """
    This function returns a moderately expander graph with `number_of_supernodes` supernodes with the
    degree of `degree_of_each_supernode` where each super node has `nodes_in_clique` nodes inside itself.
    """

    # creating a `degree_of_each_supernode`-regular network for supernodes (START)
    H = nx.random_regular_graph(d=degree_of_each_supernode, n=number_of_supernodes)
    # creating a `degree_of_each_supernode`-regular network for supernodes (END)

    # creating cliques of `nodes_in_clique` nodes (START)
    G = nx.complete_graph(n=nodes_in_clique)
    # creating cliques of `nodes_in_clique` nodes (END)

    # replacing supernodes with cliques and modifying edges (START)
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
    # replacing supernodes with cliques and modifying edges (START)
    return G


def simulation(realworld_graph=None, num_red=1, orange_p=0, k=5, dict_args={"type": TYPE_ERDOS_RENYI, "n": 200},
               dict_counter_measure={"type": COUNTER_MEASURE_NONE}, visualization=False, seed=None):
    """
    This function is the implementation of the proposed model and counter measures. The followings describe each
     parameter's use:
    realworld_graph: we use a dictionary similar to the one defined in `dataset_setup.py` to import external datasets.
                    This parameter should be equal to None in case the experiment is intended to run just on synthetic
                    networks. If both this parameter and `dict_args` are set, the simulation will run on a synthetic
                    network with characteristics similar to the given realworld_graph.
    num_red: number of red colored nodes in the first epoch of the simulation.
    orange_p: the percentage of orange colored nodes in the first epoch of the simulation.
    k: the number of epoches that takes until a red node turns orange.
    dict_args: a dictionary of the parameters to be used for creating a synthetic network (explained in `main.py`).
                If both this parameter and `realworld_graph` are set, the simulation will run on a synthetic network
                with characteristics similar to the given realworld_graph.
    dict_counter_measure: a dictionary of the parameters to be used for running the counter measure.
    visualization: set this parameter to True to save .html and .png representations in output folder
    seed: The seed value for creating pseudo-random numbers in the process.
    """
    # setting random seed value (START)
    random.seed(seed)
    # setting random seed value (END)

    # Generating the graph (START)
    if realworld_graph is not None and dict_args is not None:
        our_graph = getSNLikeGraph(realworld_graph=realworld_graph, type_graph=dict_args["type"])  # create synthetic
        # graphs similar to the real world graph
    elif realworld_graph is None and dict_args is not None:
        our_graph = getGraph(dict_args=dict_args)  # create synthetic networks
    elif realworld_graph is not None and dict_args is None:
        our_graph = getGraph(dict_args=realworld_graph)  # create realworld networks
    # Generating the graph (START)

    # generate the initial opinions of the graph
    our_graph = getInitialOpinions(graph=our_graph, num_red=num_red, orange_p=orange_p)

    #
    stop = 0
    phase = 0

    count = 0

    # assigning initial values to the variables (START)
    for node in our_graph.nodes:
        our_graph.nodes[node]['stamp'] = 0
        if dict_counter_measure["type"] == COUNTER_MEASURE_DELAYED_SPREADING:
            if our_graph.nodes[node]['vote'] == NODE_COLOR_RED:
                our_graph.nodes[node]['sleep_timer'] = dict_counter_measure["sleep_timer"]
    # assigning initial values to the variables (END)

    # counting and recording the red nodes in the network (START)
    red_nodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]
    current_num_red = len(red_nodes_initial)
    # counting and recording the red nodes in the network (END)

    # counting and recording the orange nodes in the network (START)
    orange_nodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_ORANGE]
    current_num_orange = len(orange_nodes_initial)
    # counting and recording the orange nodes in the network (END)
    #
    # counting and recording the pale green nodes in the network (START)
    current_num_pale_green = 0
    # counting and recording the pale green nodes in the network (END)

    # counting and recording the white nodes in the network (START)
    white_nodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_WHITE]
    current_num_white = len(white_nodes_initial)
    # counting and recording the white nodes in the network (END)

    # choosing green nodes for green information counter measure (START)
    current_num_green = 0
    if dict_counter_measure["type"] == COUNTER_MEASURE_GREEN_INFORMATION:
        if dict_counter_measure["high_degree_selection_strategy"]:
            # We sort the nodes according to their degrees for the purpose of selecting high degree nodes
            sorted_ = sorted(our_graph.degree, key=lambda x: x[1], reverse=True)
            r = 0
            while current_num_green < dict_counter_measure["num_green"]:
                # turn a white node to a green node
                node_ind = sorted_[r][0]
                if our_graph.nodes[node_ind]['vote'] == NODE_COLOR_WHITE:
                    our_graph.nodes[node_ind]['vote'] = NODE_COLOR_GREEN
                    current_num_green += 1
                    current_num_white -= 1
                r = r + 1
        else:
            while current_num_green < dict_counter_measure["num_green"]:
                r = random.randint(0, our_graph.number_of_nodes() - 1)
                # turn a white node to a green node
                if our_graph.nodes[r]['vote'] == NODE_COLOR_WHITE:
                    our_graph.nodes[r]['vote'] = NODE_COLOR_GREEN
                    current_num_green += 1
                    current_num_white -= 1
    # choosing green nodes for green information counter measure (END)
    # setting values and detecting communities to be used in community detection counter measure (START)

    threshold_detection = 0
    threshold_block = 0
    sum_blocked = 0
    cs = []
    if dict_counter_measure["type"] == COUNTER_MEASURE_COMMUNITY_DETECTION:
        threshold_detection = dict_counter_measure["threshold_detection"]
        threshold_block = dict_counter_measure["threshold_block"]
        cs = nx_comm.louvain_communities(our_graph)
        c_ind = 0
        for c in cs:
            for n in c:
                our_graph.nodes[n]["comm"] = c_ind
            c_ind += 1
    # setting values and detecting communities to be used in community detection counter measure (END)

    # Setting initial values for the doubt spreading counter measure (START)
    doubt_counter = 0
    negative_doubt_shift = 0.0
    positive_doubt_shift = 0.0
    doubt_spreader = []
    doubt_ls = []
    if dict_counter_measure["type"] == COUNTER_MEASURE_DOUBT_SPREADING:
        negative_doubt_shift = dict_counter_measure["negative_doubt_shift"]
        positive_doubt_shift = dict_counter_measure["positive_doubt_shift"]
        for node in our_graph.nodes():
            our_graph.nodes[node]["doubt"] = random.normalvariate(0.5, 0.16)
            our_graph.nodes[node]['origin'] = []
            if our_graph.nodes[node]["doubt"] > 1:
                our_graph.nodes[node]["doubt"] = 1
            if our_graph.nodes[node]["doubt"] < 0:
                our_graph.nodes[node]["doubt"] = 0
            doubt_ls.append(our_graph.nodes[node]["doubt"])
        # plotting doubt values of the nodes (can be disabled in case of necessity) (START)
        plt.clf()
        plt.hist(doubt_ls, bins=30)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        # plotting doubt values of the nodes (can be disabled in case of necessity) (END)
    # Setting initial values for the doubt spreading counter measure (END)

    # Arrays to keep the track of the number of nodes in different colors during the experiments (START)
    list_num_orange = [current_num_orange]
    list_num_red = [current_num_red]
    list_num_white = [current_num_white]
    list_num_green = [current_num_green]
    list_num_blocked_edges = [0]
    # Arrays to keep the track of the number of nodes in different colors during the experiments (START)

    # Visualization setup
    h = ig.Graph.from_networkx(our_graph)
    layout = h.layout("fr")
    # Visualization setup

    # Simulation loop (START)
    step = 1
    while stop != 1:
        phase = phase + 1
        # print("phase", phase)
        change = 0
        for round in range(k + 1):
            # print("round", round)
            # running experiments without any counter measures (START)
            if dict_counter_measure["type"] == COUNTER_MEASURE_NONE:
                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been red for k rounds and becomes orange
                    if our_graph.nodes[node]['stamp'] == k:
                        our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                        current_num_orange = current_num_orange + 1
                        current_num_red = current_num_red - 1
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
                            count = count + 1
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = (jaccard_sim / denom)
                            rand = random.random()
                            if rand < r:
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                change = change + 1
                                current_num_red = current_num_red + 1
                                current_num_white = current_num_white - 1
                                # print("change", change)
            # running experiments without any counter measures (END)
            # running experiments with green information spreading counter measure (START)
            elif dict_counter_measure["type"] == COUNTER_MEASURE_GREEN_INFORMATION:
                for node in [node for node in our_graph.nodes if
                             our_graph.nodes[node]['vote'] == NODE_COLOR_RED or our_graph.nodes[node][
                                 'vote'] == NODE_COLOR_GREEN]:
                    if our_graph.nodes[node]["vote"] == NODE_COLOR_GREEN:
                        # checking whether the start_time has arrived or not
                        if step >= dict_counter_measure["start_time"]:
                            our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                            if our_graph.nodes[node]['stamp'] == k:
                                our_graph.nodes[node]['vote'] = NODE_COLOR_PALE_GREEN
                                current_num_pale_green = current_num_pale_green + 1
                                current_num_green = current_num_green - 1
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
                                        r = float(jaccard_sim / denom) * dict_counter_measure["green_spreading_ratio"]
                                        rand = random.random()
                                        if rand < r:
                                            our_graph.nodes[neigh]['vote'] = NODE_COLOR_GREEN
                                            change = change + 1
                                            current_num_green = current_num_green + 1
                                            current_num_white = current_num_white - 1
                                            # print("change", change)
                    elif our_graph.nodes[node]["vote"] == NODE_COLOR_RED:
                        our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                        # the node has been red for k rounds and becomes orange
                        if our_graph.nodes[node]['stamp'] == k:
                            our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                            current_num_orange = current_num_orange + 1
                            current_num_red = current_num_red - 1
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
                                    r = (jaccard_sim / denom)
                                    rand = random.random()
                                    if rand < r:
                                        our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                        change = change + 1
                                        current_num_red = current_num_red + 1
                                        current_num_white = current_num_white - 1
            # running experiments with green information spreading counter measure (END)
            # running experiments with hear from at least two counter measure (START)
            elif dict_counter_measure["type"] == COUNTER_MEASURE_HEAR_FROM_AT_LEAST_TWO:
                hit_nodes = []
                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been red for k rounds and becomes orange
                    if our_graph.nodes[node]['stamp'] == k:
                        our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                        current_num_orange = current_num_orange + 1
                        current_num_red = current_num_red - 1
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
                            count = count + 1
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = (jaccard_sim / denom)
                            rand = random.random()
                            if rand < r:
                                our_graph.nodes[neigh]['hit_counter'] += 1
                                hit_nodes.append(neigh)

                for hit_node in hit_nodes:
                    if (our_graph.nodes[hit_node]['hit_counter'] >= 2):
                        our_graph.nodes[hit_node]['vote'] = NODE_COLOR_RED
                        change = change + 1
                        current_num_red = current_num_red + 1
                        current_num_white = current_num_white - 1
                    our_graph.nodes[hit_node]['hit_counter'] = 0
            # running experiments with hear from at least two counter measure (END)
            # running experiments with the delayed spreading counter measure (START)
            elif dict_counter_measure["type"] == COUNTER_MEASURE_DELAYED_SPREADING:
                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                    if our_graph.nodes[node]['sleep_timer'] != 0:
                        our_graph.nodes[node]['sleep_timer'] -= 1
                        continue
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been red for k rounds and becomes orange
                    if our_graph.nodes[node]['stamp'] == k:
                        our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                        current_num_orange = current_num_orange + 1
                        current_num_red = current_num_red - 1
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
                            count = count + 1
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = (jaccard_sim / denom)
                            rand = random.random()
                            if rand < r:
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                our_graph.nodes[neigh]['sleep_timer'] = dict_counter_measure["sleep_timer"]
                                change = change + 1
                                current_num_red = current_num_red + 1
                                current_num_white = current_num_white - 1
            # running experiments with the delayed spreading counter measure (END)
            # running experiments with the community detection counter measure (START)
            elif dict_counter_measure["type"] == COUNTER_MEASURE_COMMUNITY_DETECTION:
                red_nodes = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]
                maxes = []
                if len(red_nodes) >= int(our_graph.number_of_nodes() * threshold_detection):
                    red_ratio_per_community = []
                    counter = 0
                    for c in cs:
                        red_ratio_per_community.append(0)
                        for n in c:
                            if our_graph.nodes[n]['vote'] == NODE_COLOR_RED:
                                red_ratio_per_community[counter] += 1
                        red_ratio_per_community[counter] = red_ratio_per_community[counter] / len(c)

                        if red_ratio_per_community[counter] >= threshold_block:
                            maxes.append(counter)
                            for n in cs[counter]:
                                for edge_n in our_graph.edges(n):
                                    if ((edge_n[0] == n and our_graph.nodes[edge_n[1]]["comm"] != our_graph.nodes[n][
                                        "comm"])
                                            or (edge_n[1] == n and our_graph.nodes[edge_n[0]]["comm"] !=
                                                our_graph.nodes[n]["comm"])):
                                        our_graph.edges[edge_n]["blocked"] += 1
                                        sum_blocked += 1
                        counter += 1
                for node in red_nodes:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been red for k rounds and becomes orange
                    if our_graph.nodes[node]['stamp'] == k:
                        our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                        current_num_orange = current_num_orange + 1
                        current_num_red = current_num_red - 1
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
                            count = count + 1
                            denom = Decimal(2 ** (our_graph.nodes[node]['stamp']))
                            r = (jaccard_sim / denom)
                            rand = random.random()
                            if rand < r:
                                if our_graph.nodes[node]["comm"] in maxes:
                                    if our_graph.nodes[neigh]["comm"] != our_graph.nodes[node]["comm"]:
                                        print("the spread of rumor from #" + str(node) + " to #" +
                                              str(neigh) + " has been blocked because #" + str(
                                            node) + " is in the blocked community")
                                        continue
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                change = change + 1
                                current_num_red = current_num_red + 1
                                current_num_white = current_num_white - 1
            # running experiments with the community detection counter measure (END)
            # running experiments with the doubt spreading counter measure (START)
            elif dict_counter_measure["type"] == COUNTER_MEASURE_DOUBT_SPREADING:
                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_RED]:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been red for k rounds and becomes orange
                    if our_graph.nodes[node]['stamp'] == k:
                        our_graph.nodes[node]['vote'] = NODE_COLOR_ORANGE
                        current_num_orange = current_num_orange + 1
                        current_num_red = current_num_red - 1
                    else:
                        neighlist = list(our_graph.adj[node])
                        for neigh in neighlist:
                            # only consider the white neighbors these are the only ones that can be influenced
                            if our_graph.nodes[neigh]['vote'] == NODE_COLOR_WHITE:
                                neighset = set(our_graph.adj[node])
                                # manually add the nodes to their own neighborhoods
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
                                    rand2_doubt = random.random()
                                    if our_graph.nodes[neigh]['doubt'] < rand2_doubt:
                                        our_graph.nodes[neigh]['vote'] = NODE_COLOR_RED
                                        change = change + 1
                                        current_num_red = current_num_red + 1
                                        current_num_white = current_num_white - 1

                                        # reduce doubt
                                        delta_doubt = random.uniform(0, negative_doubt_shift)
                                        our_graph.nodes[neigh]['doubt'] += delta_doubt
                                        our_graph.nodes[neigh]['origin'].append((doubt_counter, delta_doubt))
                                        doubt_counter += 1
                                        doubt_spreader.append(neigh)
                                        if our_graph.nodes[neigh]['doubt'] < 0:
                                            our_graph.nodes[neigh]['doubt'] = 0
                                    else:
                                        # add doubt
                                        delta_doubt = random.uniform(0, positive_doubt_shift)
                                        our_graph.nodes[neigh]['doubt'] += delta_doubt
                                        our_graph.nodes[neigh]['origin'].append((doubt_counter, delta_doubt))
                                        doubt_counter += 1
                                        doubt_spreader.append(neigh)
                                        if our_graph.nodes[neigh]['doubt'] > 1:
                                            our_graph.nodes[neigh]['doubt'] = 1
                # spread doubts
                print("Updating doubt values")
                doubt_spreader_tmp = []
                while len(doubt_spreader) > 0:
                    d_node = doubt_spreader.pop()
                    if our_graph.nodes[d_node]["vote"] == NODE_COLOR_ORANGE:
                        continue
                    (d, delta) = our_graph.nodes[d_node]['origin'][-1]
                    for neigh in our_graph.neighbors(d_node):
                        if d in [x for (x, y) in our_graph.nodes[d_node]['origin']]:
                            continue
                        else:
                            delta_doubt = random.uniform(0, delta)
                            our_graph.nodes[neigh]['doubt'] += delta_doubt
                            our_graph.nodes[neigh]['origin'].append((d, delta_doubt))
                            doubt_spreader_tmp.append(neigh)
                doubt_spreader.extend(doubt_spreader_tmp)
                doubt_ls = []
                for node in our_graph.nodes():
                    doubt_ls.append(our_graph.nodes[node]["doubt"])
                plt.clf()
                plt.hist(doubt_ls, bins=30)
                plt.show(block=False)
                plt.pause(0.25)
                plt.close()
            # running experiments with the doubt spreading counter measure (END)
            print("round", round, "phase", phase, current_num_orange, current_num_white, current_num_red,
                  current_num_green)
            list_num_white.append(current_num_white)
            list_num_red.append(current_num_red)
            list_num_orange.append(current_num_orange)
            list_num_green.append(current_num_green)
            list_num_blocked_edges.append(sum_blocked)
            step += 1
            print("list_num_white", list_num_white)
            print("list_num_red", list_num_red)
            print("list_num_orange", list_num_orange)
            print("list_num_green", list_num_green)
            print("list_num_blocked_Edges", list_num_blocked_edges)
            if visualization:
                visual_style = {}
                visual_style["vertex_size"] = 10
                visual_style["vertex_color"] = []
                # visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
                for n in our_graph.nodes:
                    if our_graph.nodes[n]['vote'] == NODE_COLOR_ORANGE:
                        visual_style["vertex_color"].append("orange")
                    elif our_graph.nodes[n]['vote'] == NODE_COLOR_WHITE:
                        visual_style["vertex_color"].append("white")
                    elif our_graph.nodes[n]['vote'] == NODE_COLOR_GREEN:
                        visual_style["vertex_color"].append("green")
                    elif our_graph.nodes[n]['vote'] == NODE_COLOR_RED:
                        visual_style["vertex_color"].append("red")
                    elif our_graph.nodes[n]['vote'] == NODE_COLOR_PALE_GREEN:
                        visual_style["vertex_color"].append("#98FB98")
                visual_style["layout"] = layout
                visual_style["bbox"] = (1200, 1200)
                visual_style["margin"] = 10
                ig.plot(h, "Output/social_network" + str(step) + ".png",**visual_style)
        print("change", change)
        if change == 0:
            stop = 1
    # Simulation loop (END)
    for ind in range(0, len(list_num_orange)):
        list_num_orange[ind] = list_num_orange[ind] / our_graph.number_of_nodes()
        list_num_white[ind] = list_num_white[ind] / our_graph.number_of_nodes()
        list_num_red[ind] = list_num_red[ind] / our_graph.number_of_nodes()
        list_num_green[ind] = list_num_green[ind] / our_graph.number_of_nodes()
    return [list_num_white, list_num_red, list_num_orange, list_num_green, list_num_blocked_edges]
