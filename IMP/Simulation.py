# imports
import networkx as nx
import os
import random
from decimal import Decimal
from IMP import twitter_loc
import matplotlib.pyplot as plt

abs_path = os.path.abspath(os.path.dirname(__file__))

# counter measure IDs
COUNTER_MEASURE_NONE = 0
COUNTER_MEASURE_COUNTER_RUMOR_SPREAD = 1
COUNTER_MEASURE_HEAR_FROM_AT_LEAST_TWO = 2
COUNTER_MEASURE_DELAYED_SPREADING = 3
# counter measure IDs
# node color IDs
NODE_COLOR_BLACK = 1
NODE_COLOR_WHITE = -1
NODE_COLOR_GRAY = 0
NODE_COLOR_RESERVED = 2
NODE_COLOR_GREEN = 3


# node colour IDs

def KClique(j, c):
    # G = nx.complete_graph(n=j)
    # #create all the distinct disjoint cliques
    # for i in range(c-1):
    #     G = nx.disjoint_union(G,nx.complete_graph(n=j))
    # #connect them with edges to make a cycle
    # for i in range(c):
    #     G.add_edge(i, i+j)
    # #print(G)
    # #print(G.number_of_nodes())

    G = nx.ring_of_cliques(j, c)

    return G


def KCliqueExpander(j, c, d):
    # something does not work here
    KC = KClique(j, c)
    # print("KCLIQUE", KC, KC.nodes())
    # print(j*c)
    mapping = dict(zip(KC, range(0, KC.number_of_nodes() - 1)))
    # temp_graph = nx.relabel_nodes(temp_graph, mapping)
    RRG = nx.random_regular_graph(d, j * c)
    RRG = nx.relabel_nodes(RRG, mapping)
    # print("RRG", RRG, RRG.nodes())
    G = nx.compose(RRG, KC)
    return G


def GetSNlikeGraph(graph, type_graph):
    # Generating the graph
    temp_graph = nx.read_edgelist(os.path.join(abs_path, graph), create_using=nx.Graph(), nodetype=int)
    mapping = dict(zip(temp_graph, range(0, temp_graph.number_of_nodes() - 1)))
    temp_graph = nx.relabel_nodes(temp_graph, mapping)
    # print("num nodes", temp_graph.number_of_nodes())
    # print('num edges', temp_graph.number_of_edges())
    total = sum(j for i, j in list(temp_graph.degree(temp_graph.nodes)))
    av_deg = total / temp_graph.number_of_nodes()
    # print("av_deg", av_deg)
    p = total / (temp_graph.number_of_nodes() * (temp_graph.number_of_nodes() - 1))
    if type_graph == 'ER':
        # print("ER graph")
        our_graph = nx.fast_gnp_random_graph(n=temp_graph.number_of_nodes(), p=p)
    if type_graph == 'BA':
        # print("BA graph")
        our_graph = nx.barabasi_albert_graph(n=temp_graph.number_of_nodes(), m=int(av_deg))
    if type_graph == 'SN':
        our_graph = temp_graph
    return our_graph


def GetGraph(SNtype, graph, type_graph, d, dict_args):
    print("in get graph")
    if SNtype:
        our_graph = GetSNlikeGraph(graph, type_graph)
    else:
        if type_graph == "cycle":
            our_graph = nx.cycle_graph(dict_args["n"])
        if type_graph == "KClique":
            our_graph = KClique(dict_args["num_cliques"], dict_args["clique_size"])
        if type_graph == "KCliqueExpander":
            our_graph = KCliqueExpander(dict_args["num_cliques"], dict_args["clique_size"], d)
        if type_graph == "Complete":
            our_graph = nx.complete_graph(dict_args["n"])
        if type_graph == "moderatelyExpander":
            our_graph = moderatelyExpander(degree_of_each_supernode=dict_args["degree_of_each_supernode"],
                                           number_of_supernodes=dict_args["number_of_supernodes"],
                                           nodes_in_clique=dict_args["nodes_in_clique"])
        if type_graph == "LFR":
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
            our_graph = nx.LFR_benchmark_graph(n=dict_args["n"], tau1=dict_args["tau1"], tau2=dict_args["tau2"],
                                               mu=dict_args["mu"], min_degree=dict_args["min_degree"],
                                               max_degree=dict_args["max_degree"],
                                               min_community=dict_args["min_community"],
                                               max_community=dict_args["max_community"],
                                               tol=dict_args["tol"], max_iters=dict_args["max_iters"],
                                               seed=dict_args["seed"])
            print("edges: " + str(our_graph.number_of_edges()))
            print("nodes: " + str(len(our_graph.nodes())))
            nx.draw(our_graph)
            # plt.savefig("filenameLFR.png")

    return our_graph


def GetInitialOpinions(graph, num_black, gray_p):
    num_black_c = 0
    print("in get initial opinions")
    for node in graph.nodes:
        graph.nodes[node]['hit_counter'] = 0
        if (random.random() < gray_p):
            # these are the nodes that will have vote gray and thus are not affected by the neighbours
            graph.nodes[node]['vote'] = NODE_COLOR_GRAY
        else:
            # the nodes that will have vote black or white
            graph.nodes[node]['vote'] = NODE_COLOR_WHITE
    while (num_black_c < num_black):
        r_node = random.randint(0, graph.number_of_nodes() - 1)
        if (graph.nodes[r_node]['vote'] == NODE_COLOR_WHITE):
            graph.nodes[r_node]['vote'] = NODE_COLOR_BLACK
            num_black_c += 1

    return graph


def moderatelyExpander(degree_of_each_supernode, number_of_supernodes, nodes_in_clique):
    H = nx.random_regular_graph(d=degree_of_each_supernode, n=number_of_supernodes)

    G = nx.complete_graph(n=nodes_in_clique)
    H_nodes = list(H.nodes())

    print("nodes : " + str(H_nodes))
    for i in range(len(H_nodes) - 1):
        G = nx.disjoint_union(G, nx.complete_graph(n=nodes_in_clique))
    for i in H_nodes:
        edges_i = list(H.edges(i))
        print("edges in " + str(i))
        print(edges_i)
        for j in range(len(edges_i)):
            print(str(edges_i[j]) + " => (" + str(edges_i[j][0] * nodes_in_clique) + ", " + str(
                edges_i[j][1] * nodes_in_clique) + ")")
            G.add_edge(
                random.randint(edges_i[j][0] * nodes_in_clique, edges_i[j][0] * nodes_in_clique + nodes_in_clique - 1),
                random.randint(edges_i[j][1] * nodes_in_clique, edges_i[j][1] * nodes_in_clique + nodes_in_clique - 1))
        H.remove_node(i)

    nx.draw(G)
    plt.savefig("filename.png")
    return G


def Simulation(graph, SNtype, type_graph, num_black, gray_p, tresh, d, dict_args, k,
               dict_counter_measure={"id": COUNTER_MEASURE_NONE}):
    # generate the graph
    # dict_args is used for the purpose of passing multiple arguments for the generation of LFR networks.
    our_graph = GetGraph(graph=graph, SNtype=SNtype, type_graph=type_graph, d=d, dict_args=dict_args)

    # generate the initial opinions of the graph
    our_graph = GetInitialOpinions(graph=our_graph, num_black=num_black, gray_p=gray_p)

    # now we consider the update rule
    stop = 0
    phase = 0
    round = 1
    sum_jaccard_sim = 0
    count = 0

    # initialize the field temp_vote and stamp
    # note that -5 is just a placeholder
    for node in our_graph.nodes:
        # our_graph.nodes[node]['temp_vote'] = -5
        our_graph.nodes[node]['stamp'] = 0

    blacknodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_BLACK]
    cur_num_black = len(blacknodes_initial)

    gray_nodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_GRAY]
    cur_num_gray = len(gray_nodes_initial)

    whitenodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_WHITE]
    cur_num_white = len(whitenodes_initial)

    green_nodes_initial = []
    cur_num_green = 0
    list_num_green = []
    if (dict_counter_measure["id"] == COUNTER_MEASURE_COUNTER_RUMOR_SPREAD):
        while (cur_num_green < dict_counter_measure["num_green"]):
            r = random.randint(0, our_graph.number_of_nodes())
            # turn a white node to a green node
            if (our_graph.nodes[r]['vote'] == NODE_COLOR_WHITE):
                our_graph.nodes[r]['vote'] = NODE_COLOR_GREEN
                green_nodes_initial = [r]
                cur_num_green += 1
                cur_num_white -= 1

    list_num_gray = [cur_num_gray]
    list_num_black = [cur_num_black]
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
                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_BLACK]:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been black for k rounds and becomes gray
                    if our_graph.nodes[node]['stamp'] == k:
                        # print("becoming gray", node)
                        our_graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                        cur_num_gray = cur_num_gray + 1
                        cur_num_black = cur_num_black - 1
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
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_BLACK
                                change = change + 1
                                cur_num_black = cur_num_black + 1
                                cur_num_white = cur_num_white - 1
                                # print("change", change)
            elif (dict_counter_measure["id"] == COUNTER_MEASURE_COUNTER_RUMOR_SPREAD):
                if (step >= dict_counter_measure["start_time"]):
                    for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_GREEN]:
                        our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                        # the node has been green for k rounds and becomes gray
                        if our_graph.nodes[node]['stamp'] == k:
                            # print("becoming gray", node)
                            our_graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                            cur_num_gray = cur_num_gray + 1
                            cur_num_green = cur_num_green - 1
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
                                    our_graph.nodes[neigh]['vote'] = NODE_COLOR_GREEN
                                    change = change + 1
                                    cur_num_green = cur_num_green + 1
                                    cur_num_white = cur_num_white - 1
                                    # print("change", change)
                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_BLACK]:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been black for k rounds and becomes gray
                    if our_graph.nodes[node]['stamp'] == k:
                        # print("becoming gray", node)
                        our_graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                        cur_num_gray = cur_num_gray + 1
                        cur_num_black = cur_num_black - 1
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
                                our_graph.nodes[neigh]['vote'] = NODE_COLOR_BLACK
                                change = change + 1
                                cur_num_black = cur_num_black + 1
                                cur_num_white = cur_num_white - 1
                                # print("change", change)
            elif (dict_counter_measure["id"] == COUNTER_MEASURE_HEAR_FROM_AT_LEAST_TWO):
                hit_nodes=[]
                for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == NODE_COLOR_BLACK]:
                    our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                    # the node has been black for k rounds and becomes gray
                    if our_graph.nodes[node]['stamp'] == k:
                        # print("becoming gray", node)
                        our_graph.nodes[node]['vote'] = NODE_COLOR_GRAY
                        cur_num_gray = cur_num_gray + 1
                        cur_num_black = cur_num_black - 1
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
                        our_graph.nodes[hit_node]['vote'] = NODE_COLOR_BLACK
                        change = change + 1
                        cur_num_black = cur_num_black + 1
                        cur_num_white = cur_num_white - 1
                    our_graph.nodes[hit_node]['hit_counter']=0

                            # print("change", change)
            print("round", round, "phase", phase, cur_num_gray, cur_num_white, cur_num_black, cur_num_green)
            list_num_white.append(cur_num_white)
            list_num_black.append(cur_num_black)
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
