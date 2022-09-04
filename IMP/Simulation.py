#imports
import networkx as nx
import os
import random
from decimal import Decimal
from IMP import twitter_loc
abs_path = os.path.abspath(os.path.dirname(__file__))

def KClique(j,c):
    G = nx.complete_graph(n=j)
    #create all the distinct disjoint cliques
    for i in range(c-1):
        G = nx.disjoint_union(G,nx.complete_graph(n=j))
    #connect them with edges to make a cycle
    for i in range(c):
        G.add_edge(i, i+j)
    #print(G)
    #print(G.number_of_nodes())
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

def GetGraph(SNtype, graph, type_graph, j,c,d):
    print("in get graph")
    if SNtype:
        our_graph = GetSNlikeGraph(graph, type_graph)
    else:
        if type_graph == "cycle":
            our_graph = nx.cycle_graph(c)
        if type_graph == "KClique":
            our_graph = KClique(j,c)
        if type_graph == "KCliqueExpander":
            our_graph = KCliqueExpander(j,c,d)
        if type_graph == "Complete":
            our_graph = nx.complete_graph(c)
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





def Simulation(graph, SNtype, type_graph,p, gray_p,k, tresh, d, j, c):
    #generate the graph
    our_graph = GetGraph(graph=graph, SNtype=SNtype, type_graph=type_graph, d=d, j=j, c=c)

    #generate the initial opinions of the graph
    our_graph = GetInitialOpinions(graph=our_graph, p=p, gray_p=gray_p)

    #now we consider the update rule
    stop = 0
    phase = 0
    round = 1

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













    return [list_num_white, list_num_black, list_num_gray]

