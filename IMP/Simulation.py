#imports
import networkx as nx
import os
import random
from decimal import Decimal
from IMP import twitter_loc
import matplotlib.pyplot as plt
abs_path = os.path.abspath(os.path.dirname(__file__))

def KClique(j,c):
    # G = nx.complete_graph(n=j)
    # #create all the distinct disjoint cliques
    # for i in range(c-1):
    #     G = nx.disjoint_union(G,nx.complete_graph(n=j))
    # #connect them with edges to make a cycle
    # for i in range(c):
    #     G.add_edge(i, i+j)
    # #print(G)
    # #print(G.number_of_nodes())


    G=nx.ring_of_cliques(j,c)


    return G

def KCliqueExpander(j,c,d):
    #something does not work here
    KC = KClique(j,c)
    #print("KCLIQUE", KC, KC.nodes())
    #print(j*c)
    mapping = dict(zip(KC, range(0, KC.number_of_nodes() - 1)))
    #temp_graph = nx.relabel_nodes(temp_graph, mapping)
    RRG = nx.random_regular_graph(d, j*c)
    RRG = nx.relabel_nodes(RRG, mapping)
    #print("RRG", RRG, RRG.nodes())
    G = nx.compose(RRG, KC)
    return G

def GetSNlikeGraph(graph, type_graph):
    # Generating the graph
    temp_graph = nx.read_edgelist(os.path.join(abs_path, graph), create_using=nx.Graph(), nodetype=int)
    mapping = dict(zip(temp_graph, range(0, temp_graph.number_of_nodes() - 1)))
    temp_graph = nx.relabel_nodes(temp_graph, mapping)
    #print("num nodes", temp_graph.number_of_nodes())
    #print('num edges', temp_graph.number_of_edges())
    total = sum(j for i, j in list(temp_graph.degree(temp_graph.nodes)))
    av_deg = total / temp_graph.number_of_nodes()
    #print("av_deg", av_deg)
    p = total / (temp_graph.number_of_nodes() * (temp_graph.number_of_nodes() - 1))
    if type_graph == 'ER':
        #print("ER graph")
        our_graph = nx.fast_gnp_random_graph(n=temp_graph.number_of_nodes(), p=p)
    if type_graph == 'BA':
        #print("BA graph")
        our_graph = nx.barabasi_albert_graph(n=temp_graph.number_of_nodes(), m=int(av_deg))
    if type_graph == 'SN':
        our_graph = temp_graph
    return our_graph

def GetGraph(SNtype, graph, type_graph, d,dict_args):
    print("in get graph")
    if SNtype:
        our_graph = GetSNlikeGraph(graph, type_graph)
    else:
        if type_graph == "cycle":
            our_graph = nx.cycle_graph(c)
        if type_graph == "KClique":
            our_graph = KClique(dict_args["num_cliques"],dict_args["clique_size"])
        if type_graph == "KCliqueExpander":
            our_graph = KCliqueExpander(dict_args["num_cliques"],dict_args["clique_size"],d)
        if type_graph == "Complete":
            our_graph = nx.complete_graph(c)
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
    return our_graph

def GetInitialOpinions(graph, p, gray_p):
    print("in get initial opinions")
    for node in graph.nodes:
        if (random.random() < gray_p):
            #these are the nodes that will have vote gray and thus are not affected by the neighbours
            graph.nodes[node]['vote'] = 0
        else:
            #the nodes that will have vote black or white
            graph.nodes[node]['vote'] = 2
    for node in [node for node in graph.nodes if graph.nodes[node]['vote'] == 2]:
        if (random.random() < p):
            graph.nodes[node]['vote'] = 1
        else:
            graph.nodes[node]['vote'] = -1


    return graph

def moderatelyExpander(degree_of_each_supernode,number_of_supernodes,nodes_in_clique):
    H=nx.random_regular_graph(d=degree_of_each_supernode,n=number_of_supernodes)

    G=nx.complete_graph(n=nodes_in_clique)
    H_nodes=list(H.nodes())

    print("nodes : "+str(H_nodes))
    for i in range(len(H_nodes)-1):
        G = nx.disjoint_union(G, nx.complete_graph(n=nodes_in_clique))
    for i in H_nodes:
        edges_i = list(H.edges(i))
        print("edges in " + str(i))
        print(edges_i)
        for j in range(len(edges_i)):
            print(str(edges_i[j])+ " => ("+str(edges_i[j][0]*nodes_in_clique)+", "+str(edges_i[j][1]*nodes_in_clique)+")")
            G.add_edge(edges_i[j][0]*nodes_in_clique, edges_i[j][1]*nodes_in_clique)
        H.remove_node(i)

    nx.draw(G)
    plt.savefig("filename.png")
    return G



def Simulation(graph, SNtype, type_graph,p, gray_p, tresh, d, dict_args,k):
    #generate the graph
    # dict_args is used for the purpose of passing multiple arguments for the generation of LFR networks.
    our_graph = GetGraph(graph=graph, SNtype=SNtype, type_graph=type_graph, d=d, dict_args=dict_args)

    #generate the initial opinions of the graph
    our_graph = GetInitialOpinions(graph=our_graph, p=p, gray_p=gray_p)

    #now we consider the update rule
    stop = 0
    phase = 0
    round = 1
    sum_jaccard_sim = 0
    count = 0

    #initialize the field temp_vote and stamp
    #note that -5 is just a placeholder
    for node in our_graph.nodes:
        #our_graph.nodes[node]['temp_vote'] = -5
        our_graph.nodes[node]['stamp'] = 0

    blacknodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == 1]
    cur_num_black = len(blacknodes_initial)
    list_num_black = [cur_num_black]

    gray_nodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == 0]
    cur_num_gray = len(gray_nodes_initial)
    list_num_gray = [cur_num_gray]



    whitenodes_initial = [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == -1]
    cur_num_white = len(whitenodes_initial)
    list_num_white = [cur_num_white]


    #now (in contrast to the prev project) we proceed in phases, where each phase consists of k rounds.
    while stop != 1:
        phase = phase + 1
        #print("phase", phase)
        change = 0
        for round in range(k+1):
            #print("round", round)
            for node in [node for node in our_graph.nodes if our_graph.nodes[node]['vote'] == 1]:
                our_graph.nodes[node]['stamp'] = our_graph.nodes[node]['stamp'] + 1
                #the node has been black for k rounds and becomes gray
                if our_graph.nodes[node]['stamp'] == k:
                    #print("becoming gray", node)
                    our_graph.nodes[node]['vote'] = 0
                    cur_num_gray = cur_num_gray + 1
                    cur_num_black = cur_num_black - 1
                else:
                    #print("node", node)
                    neighlist = list(our_graph.adj[node])
                    #print("neighlist",neighlist)
                    #only consider the white neighbors these are the only ones that can be influenced
                    for neigh in [neigh for neigh in neighlist if our_graph.nodes[neigh]['vote'] == -1]:
                        #manually add the nodes to their own neighborhoods
                        neighset = set(our_graph.adj[node])
                        neighset.add(node)
                        neighsetneigh = set(our_graph.adj[neigh])
                        neighsetneigh.add(neigh)
                        intersection_neigh = neighset.intersection(neighsetneigh)
                        union_neigh = neighset.union(neighsetneigh)
                        jaccard_sim = Decimal(len(intersection_neigh) / len(union_neigh))
                        sum_jaccard_sim += jaccard_sim
                        count = count + 1
                        #print("jaccard_sim", jaccard_sim)
                        denom = Decimal(2**(our_graph.nodes[node]['stamp']))
                        r = (jaccard_sim / denom)
                        rand = random.random()
                        #print("r and rand", r, rand)
                        if rand < r:
                            #print("neigh",neigh) #see if there are duplicates
                            our_graph.nodes[neigh]['vote'] = 1
                            change = change + 1
                            cur_num_black = cur_num_black + 1
                            cur_num_white = cur_num_white - 1
                            #print("change", change)

            print("round", round, "phase", phase, cur_num_gray, cur_num_white, cur_num_black)
            list_num_white.append(cur_num_white)
            list_num_black.append(cur_num_black)
            list_num_gray.append(cur_num_gray)
            print("listnumwhite", list_num_white)
            print("listnumblack", list_num_black)
            print("listnumgray", list_num_gray)
        print("change", change)
        if change == 0:
            stop = 1












    print("av jaccard", sum_jaccard_sim/count)
    return [list_num_white, list_num_black, list_num_gray]

