# %%
# (START) imports
from ast import Try
import sys
import os
import pathlib
from typing import Type
import networkx
import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stdlib_python import stdio
from IPython import display
# (END) imports
# (START) parameters
network_type = 1  # 1: Cycle, 2: Complete, 3: Erdos-Renyi
number_of_nodes = 500
initial_infected_nodes = 1
days_infected = 5
# -1: Fully immune, 0,1,2,...: the number of days that a node stays immune
immunity_days = -1
erdos_renyi_p = 0.4
# 0: when full immunity is acheived, 1: when a pandemic happens, 2: when a pandemic is over
condition_to_stop = 0
definition_of_pandemic = 0.1
simulation_count = 100
# 1: Just print the reports and save them in the output folder, 2: Just print the reports and save the reports and figures in the output folder, 3:  print the reports, show the figures and save them in the output folder 
visualization_type = 2
# (END) parameters
# (START) variables
infection_days_array = list()
immunity_days_array = list()
infection_state_array = list()
infected_count = 0
immune_count = 0
# (END) variables
# (START) Constants
CUR_DIR = os.getcwd()
RES_DIR = CUR_DIR+"/Results"
# (END) Constants


def get_params():
    ''' 
    this function receives user's inputs and keeps them in their respective variables.

    inputs with their default values:

        network_type = 1

        number_of_nodes = 500

        initial_infected_nodes = 1

        days_infected = 5

        immunity_days = -1

        condition_to_stop = 0

        definition_of_pandemic = 0.1

        simulation_count = 100

        visualization_type = 0

        erdos_renyi_p = 0.4

    '''

    global network_type
    global number_of_nodes
    global initial_infected_nodes
    global days_infected
    global immunity_days
    global condition_to_stop
    global definition_of_pandemic
    global simulation_count
    global visualization_type
    global erdos_renyi_p

    # (START) reading network type
    while(True):
        stdio.writeln("Choose a type of network: ")
        stdio.writeln("1: Cycle, 2: Complete, 3: Erdos-Renyi ")
        network_type = input(
            "Enter the type of network [The default value is 1]: ")
        if(network_type == ''):
            network_type = 1
            break
        if(not network_type.isnumeric()):
            stdio.writeln("! Please enter a number")
            stdio.writeln(
                "=========================================================")
            continue
        network_type = int(network_type)
        if(network_type >= 1 and network_type <= 3):
            if(network_type == 3):  # if it's erdos renyi
                while(True):
                    stdio.writeln("The probability of edge creation: ")
                    erdos_renyi_p = input(
                        "Enter the probability of edge creation in erdos-renyi networks : [The default value is 0.4]:")
                    if(erdos_renyi_p == ''):
                        erdos_renyi_p = 0.4
                        break
                    try:
                        erdos_renyi_p = float(erdos_renyi_p)
                    except:
                        stdio.writeln("! Please enter a float number")
                        stdio.writeln(
                            "=========================================================")
                        continue
                    if(erdos_renyi_p <= 0.0 or erdos_renyi_p >= 1.0):
                        stdio.writeln("! Enter a value between 0 and 1")
                        stdio.writeln(
                            "=========================================================")
                    else:
                        break
            break
        else:
            stdio.writeln("! Please enter a correct number")
            stdio.writeln(
                "=========================================================")
    # (END) reading network type
    stdio.writeln(
        "---------------------------------------------------------")
    # (START) reading the number of nodes
    while(True):
        number_of_nodes = input(
            "Enter the number of nodes in the network [The default value is 500]: ")
        if(number_of_nodes == ''):
            number_of_nodes = 500
            break
        if(not number_of_nodes.isnumeric()):
            stdio.writeln("! Please enter a number")
            stdio.writeln(
                "=========================================================")
            continue
        number_of_nodes = int(number_of_nodes)
        break
    # (END) reading the number of nodes
    stdio.writeln(
        "---------------------------------------------------------")
    # (START) reading the number of initially infected nodes
    while(True):
        initial_infected_nodes = input(
            "Enter the number of nodes that are infected in the first step: [The default value is 1]: ")
        if(initial_infected_nodes == ''):
            initial_infected_nodes = 1
            break
        if(not initial_infected_nodes.isnumeric()):
            stdio.writeln("! Please enter a number")
            stdio.writeln(
                "=========================================================")
            continue
        initial_infected_nodes = int(initial_infected_nodes)
        if(initial_infected_nodes <= number_of_nodes):
            break
        else:
            stdio.writeln(
                "! The number of infected nodes should be smaller than the number of nodes in the network ("+str(number_of_nodes)+")")
            stdio.writeln(
                "=========================================================")
    # (END) reading the number of initially infected nodes
    stdio.writeln(
        "---------------------------------------------------------")
    # (START) reading the number of days that a node stays infected
    while(True):
        days_infected = input(
            "Enter the number of days that a node stays infected: [The default value is 5]: ")
        if(days_infected == ''):
            days_infected = 5
            break
        if(not days_infected.isnumeric()):
            stdio.writeln("! Please enter a number")
            stdio.writeln(
                "=========================================================")
            continue
        days_infected = int(days_infected)
        break
    # (END) reading the number of days that a node stays infected
    stdio.writeln(
        "---------------------------------------------------------")
    # (START) reading immunity situation
    while(True):
        stdio.writeln("Choose an immunity condition: ")
        stdio.writeln(
            "-1: Fully immune, 0,1,2,...: the number of days that a node stays immune ")
        immunity_days = input(
            "Enter the immunity condition [The default value is -1]: ")
        if(immunity_days == ''):
            immunity_days = -1
            break
        if(not immunity_days.isnumeric()):
            stdio.writeln("! Please enter a number")
            stdio.writeln(
                "=========================================================")
            continue
        immunity_days = int(immunity_days)
        if(immunity_days >= -1):
            break
        else:
            stdio.writeln("! Please enter a correct number")
            stdio.writeln(
                "=========================================================")
    # (END) reading network type
    stdio.writeln(
        "---------------------------------------------------------")
    # (START) reading the condition to stop
    while(True):
        stdio.writeln("Choose a condition to stop the process: ")
        stdio.writeln(
            "0: when full immunity is acheived, 1: when a pandemic happens, 2: when a pandemic is over")
        condition_to_stop = input(
            "Enter the stop condition : [The default value is 0]:")
        if(condition_to_stop == ''):
            condition_to_stop = 0
            break
        if(not condition_to_stop.isnumeric()):
            stdio.writeln("! Please enter a number")
            stdio.writeln(
                "=========================================================")
            continue
        condition_to_stop = int(condition_to_stop)
        if(condition_to_stop == 0 and immunity_days != -1):
            stdio.writeln(
                "! You can't choose 0 because of your previous choice")
            stdio.writeln(
                "=========================================================")
        else:
            break
    # (END) reading the condition to stop
    stdio.writeln(
        "---------------------------------------------------------")
    # (START) reading the definition of the pandemic
    if(condition_to_stop != 0):
        while(True):
            stdio.writeln("What is your definition of a pandemic: ")
            definition_of_pandemic = input(
                "Enter the proportion of infected nodes based on your definition means a pandemic : [The default value is 0.1]:")
            if(definition_of_pandemic == ''):
                definition_of_pandemic = 0.1
                break

            try:
                definition_of_pandemic = float(definition_of_pandemic)
            except:
                stdio.writeln("! Please enter a float number")
                stdio.writeln(
                    "=========================================================")
                continue
            if(definition_of_pandemic <= 0.0 or definition_of_pandemic >= 1.0):
                stdio.writeln("! Enter a value between 0 and 1")
                stdio.writeln(
                    "=========================================================")
            else:
                break
    # (END) reading the definition of the pandemic
    stdio.writeln(
        "---------------------------------------------------------")
    # (START) reading the simulation count
    while(True):
        simulation_count = input(
            "How many times should the simulation be repeated? [The default value is 100]: ")
        if(simulation_count == ''):
            simulation_count = 100
            break
        if(not simulation_count.isnumeric()):
            stdio.writeln("! Please enter a number")
            stdio.writeln(
                "=========================================================")
            continue
        simulation_count = int(simulation_count)
        break
    # (END) reading the simulation count
    stdio.writeln(
        "---------------------------------------------------------")
    # (START) reading the visualization type
    while(True):
        stdio.writeln("Choose a type of visualization: ")
        stdio.writeln("1: Just print the reports and save them in the output folder, 2: Just print the reports and save the reports and figures in the output folder, 3:  print the reports, show the figures and save them in the output folder ")
        visualization_type = input(
            "Enter the type of visualization [The default value is 2]: ")

        if(visualization_type == ''):
            visualization_type = 2
            break
        if(not visualization_type.isnumeric()):
            stdio.writeln("! Please enter a number")
            stdio.writeln(
                "=========================================================")
            continue
        visualization_type = int(visualization_type)
        if(visualization_type >= 1 and visualization_type <= 3):
            break
        else:
            stdio.writeln("! Please enter a correct number")
            stdio.writeln(
                "=========================================================")
    # (END) reading the visualization type
    stdio.writeln(
        "---------------------------------------------------------")


def print_params():
    '''
    This function prints the names' and the values' of the parameters
    '''
    global network_type
    global number_of_nodes
    global initial_infected_nodes
    global days_infected
    global immunity_days
    global condition_to_stop
    global definition_of_pandemic
    global simulation_count
    global visualization_type
    global erdos_renyi_p

    stdio.writeln("network_type = "+str(network_type))
    stdio.writeln("number_of_nodes = "+str(number_of_nodes))
    stdio.writeln("initial_infected_nodes ="+str(initial_infected_nodes))
    stdio.writeln("days_infected = "+str(days_infected))
    stdio.writeln("immunity_days = "+str(immunity_days))
    stdio.writeln("condition_to_stop = "+str(condition_to_stop))
    stdio.writeln("definition_of_pandemic = "+str(definition_of_pandemic))
    stdio.writeln("simulation_count = "+str(simulation_count))
    stdio.writeln("visualization_type = "+str(visualization_type))
    stdio.writeln("erdos_renyi_p = "+str(erdos_renyi_p))


def create_dir():
    global RES_DIR    
    # (START) create results directory
    res_path = pathlib.Path(RES_DIR)
    if(not res_path.exists()):
        res_path.mkdir()

    tmp = 0
    for d in res_path.iterdir():        
        tmp += 1
    RES_DIR += "/"+str(tmp)
    os.mkdir(RES_DIR)
    # (END) create results directory


def __main__():

    global network_type
    global number_of_nodes
    global initial_infected_nodes
    global days_infected
    global immunity_days
    global condition_to_stop
    global definition_of_pandemic
    global simulation_count
    global visualization_type
    global erdos_renyi_p
    global infection_state_array
    global infection_days_array
    global immunity_days_array
    global infected_count
    global immune_count
    

    create_dir()
    get_params()
    print_params()

    G = None  # graph object

    # create object
    if(network_type == 1):
        G = networkx.cycle_graph(number_of_nodes)
    elif(network_type == 2):
        G = networkx.complete_graph(number_of_nodes)
    else:
        G = networkx.erdos_renyi_graph(number_of_nodes, erdos_renyi_p)
    
    pos = networkx.spring_layout(G,k=1/number_of_nodes)
    
    days=list()
    pandemic_state=list()
    
    for s in range(simulation_count):
        print("Simulation #"+str(s))
        # (START) assigning initial values to infection arrays
        infection_state_array = [0]*number_of_nodes
        infection_days_array = [0]*number_of_nodes
        immunity_days_array = [0]*number_of_nodes
        # (END) assigning initial values to infection arrays
        # (START) generate random numbers to select initially infected nodes
        for i in range(initial_infected_nodes):
            r = random.randint(0, number_of_nodes-1)                        
            infection_state_array[r] = 1
        # (END) generate random numbers to select initially infected nodes
        os.mkdir(RES_DIR+"/"+str(s))        
        counter=0
        immune_data=list()
        infection_data=list()
        vulnerable_data=list()
        pandemic_occured=False
        while(True):
            print("..... Day #"+str(counter))
            if(counter>0):
                tmp_infection_state_array=infection_state_array.copy();
                for n in range(number_of_nodes):
                    if(infection_state_array[n]==0):
                        r=random.randint(0,len(list(G.neighbors(n)))-1)
                        if(infection_state_array[list(G.neighbors(n))[r]]==1):
                            tmp_infection_state_array[n]=1                    
                infection_state_array=tmp_infection_state_array.copy()
            
            infected_count = 0
            immune_count = 0
            for j in range(number_of_nodes):
                if(infection_state_array[j] == 2):
                    immune_count += 1
                    immunity_days_array[j] += 1
                    if(immunity_days_array[j] == immunity_days):
                        immunity_days_array[j] = 0
                        infection_state_array[j] = 0
                        immune_count -= 1

                if(infection_state_array[j] == 1):
                    infected_count += 1
                    infection_days_array[j] += 1
                    if(infection_days_array[j] == days_infected):
                        infection_days_array[j] = 0
                        infection_state_array[j] = 2
                        infected_count -= 1
                        immune_count += 1

            infection_data.append(infected_count)
            immune_data.append(immune_count)
            vulnerable_data.append(number_of_nodes-immune_count-infected_count)
            # illustrate graph
            if(visualization_type==2 or visualization_type==3):
                f=plt.figure()
                
                node_colors=[0]*number_of_nodes
                for c in range(number_of_nodes):
                    if(infection_state_array[c]==0):
                        node_colors[c]='blue'
                    elif(infection_state_array[c]==1):
                        node_colors[c]='red'
                    elif(infection_state_array[c]==2):
                        node_colors[c]='green'
                        
                
                networkx.draw(G, node_color=node_colors,ax=f.add_subplot(221),pos=pos)
                
                f.add_subplot(222)
                plt.title("Neutral nodes")
                plt.plot(vulnerable_data)
                f.add_subplot(223)
                plt.title("Infected nodes")
                plt.plot(infection_data)
                f.add_subplot(224)
                plt.title("Immune nodes")
                plt.plot(immune_data)
                
                plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
                f.savefig(RES_DIR+"/"+str(s)+"/"+str(counter)+".png")
                if(visualization_type==3):
                    display.clear_output(wait=True)
                    display.display(f)                    
                    time.sleep(1.0)   
                plt.close(f)         
            counter+=1
            if(infected_count/number_of_nodes >= definition_of_pandemic):
                pandemic_occured=True                               
            # (START) checking the stop condition
            if(condition_to_stop == 0):  # when full immunity is reached
                if(immune_count == number_of_nodes or infected_count==0 ):
                    break
            elif(condition_to_stop == 1):  # when a pandemic happens
                if(infected_count/number_of_nodes >= definition_of_pandemic or infected_count==0):
                    break
            else:  # when a pandemic is over
                if(infected_count==0):
                   break 
                if(infected_count/number_of_nodes >= definition_of_pandemic):
                    pandemic_occured=True
                else:
                    if(pandemic_occured):
                        break                        
            # (END) checking the stop condition          
        file = open(RES_DIR+"/"+str(s)+"/var.txt", "a")
        file.write("vulnerable_data = "+repr(vulnerable_data) +"\n\r")
        file.write("infection_data = "+repr(infection_data) +"\n\r")
        file.write("immune_data = "+repr(immune_data)+"\n\r")
        file.write("days = "+repr(counter)+"\n\r")
        file.close()
        
        days.append(counter)
        pandemic_state.append(pandemic_occured)
        print(pandemic_state)
        
    file = open(RES_DIR+"/var.txt", "a")
    file.write("days = "+repr(days)+"\n\r")
    file.write("pandemic_state = "+repr(pandemic_state)+"\n\r")
    
    return 0


if __name__ == "__main__":
    __main__()

    # %%
