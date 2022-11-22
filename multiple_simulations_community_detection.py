"""
The code and instructions for running the simulations can be found in this file.
"""
# imports (START)
import math
import sys
import os
import matplotlib.pyplot as plt
from Simulation import *
from Utility.dataset_setup import facebook, twitter, slashdot, pokec

# from averaging import averaging
# from averaging import *

# imports (END)


# Constant referring to the directory of the code (START)
ABS_PATH = os.path.abspath(os.path.dirname(__file__))
# Constant referring to the directory of the code (END)

if __name__ == "__main__":

    # counter measure args (START)
    dict_counter_measure_community_detection = {"type": COUNTER_MEASURE_COMMUNITY_DETECTION,
                                                "threshold_detection": 0.01,
                                                "threshold_block": 0.05}
    # counter measure args (END)

    # running the simulation (START)
    avgs = dict()
    for z in range(1, 6):
        avg_list_ratio_white = [0]
        avg_list_ratio_red = [0]
        avg_list_ratio_orange = [0]
        avg_list_ratio_green = [0]

        for t in range(1, 11):
            dict_counter_measure_community_detection["threshold_detection"] = float(z / 100)
            [list_ratio_white, list_ratio_red, list_ratio_orange, list_ratio_green] = \
                simulation(realworld_graph=twitter, num_red=1, orange_p=0,
                           k=5, dict_args=None, dict_counter_measure=
                           dict_counter_measure_community_detection, seed=None)

            with open("Output/output-threshold_detection = " +
                      str(dict_counter_measure_community_detection["threshold_detection"]) + ".txt", "a") as f:
                f.write("list_num_white = " + repr(list_ratio_white) + "\n")
                f.write("list_num_red = " + repr(list_ratio_red) + "\n")
                f.write("list_num_orange = " + repr(list_ratio_orange) + "\n")
                f.write("list_num_green = " + repr(list_ratio_green) + "\n")
                f.write("------------------------------------------- \n")

            if len(avg_list_ratio_white) <= len(list_ratio_white):
                avg_list_ratio_white.extend([0] * (abs(len(avg_list_ratio_white) - len(list_ratio_white))))
            else:
                list_ratio_white.extend([list_ratio_white[-1]] * (abs(len(avg_list_ratio_white) - len(list_ratio_white))))

            if len(avg_list_ratio_red) <= len(list_ratio_red):
                avg_list_ratio_red.extend([0] * (abs(len(avg_list_ratio_red) - len(list_ratio_red))))
            else:
                list_ratio_red.extend([list_ratio_red[-1]] * (abs(len(avg_list_ratio_white) - len(list_ratio_red))))

            if len(avg_list_ratio_orange) <= len(list_ratio_orange):
                avg_list_ratio_orange.extend([0] * (abs(len(avg_list_ratio_orange) - len(list_ratio_orange))))
            else:
                list_ratio_orange.extend([list_ratio_orange[-1]] * (abs(len(avg_list_ratio_white) - len(list_ratio_orange))))

            if len(avg_list_ratio_green) <= len(list_ratio_green):
                avg_list_ratio_green.extend([0] * (abs(len(avg_list_ratio_green) - len(list_ratio_green))))
            else:
                list_ratio_green.extend([list_ratio_green[-1]] * (abs(len(avg_list_ratio_white) - len(list_ratio_green))))

            avg_list_ratio_white = [sum(i) for i in zip(avg_list_ratio_white, list_ratio_white)]
            avg_list_ratio_red = [sum(i) for i in zip(avg_list_ratio_red, list_ratio_red)]
            avg_list_ratio_orange = [sum(i) for i in zip(avg_list_ratio_orange, list_ratio_orange)]
            avg_list_ratio_green = [sum(i) for i in zip(avg_list_ratio_green, list_ratio_green)]

        avg_list_ratio_white = [i / 10 for i in avg_list_ratio_white]
        avg_list_ratio_red = [i / 10 for i in avg_list_ratio_red]
        avg_list_ratio_orange = [i / 10 for i in avg_list_ratio_orange]
        avg_list_ratio_green = [i / 10 for i in avg_list_ratio_green]

        with open("Output/output-threshold_detection = " +
                  str(dict_counter_measure_community_detection["threshold_detection"]) + ".txt", "a") as f:
            f.write("avg_list_num_white = " + repr(avg_list_ratio_white) + "\n")
            f.write("avg_list_num_red = " + repr(avg_list_ratio_red) + "\n")
            f.write("avg_list_num_orange = " + repr(avg_list_ratio_orange) + "\n")
            f.write("avg_list_num_green = " + repr(avg_list_ratio_green) + "\n")
            f.write("------------------------------------------- \n")
        plt.clf()
        plt.xlabel("rounds", fontdict=None, labelpad=None)
        plt.ylabel("the fraction of orange nodes", fontdict=None, labelpad=None)
        # plt.plot(list_num_white, "blue", label="white")
        # plt.plot(list_num_red, "red", label="red")
        plt.plot(avg_list_ratio_orange, "orange", label="orange nodes in " + twitter["name"])
        plt.title("Average ratios of 10 experiment: \n " +
                  "threshold_detection = " + str(dict_counter_measure_community_detection["threshold_detection"]))
        # plt.plot(list_num_green, "green", label="green")
        plt.legend()
        plt.savefig("Output/output-threshold_detection = " +
                    str(dict_counter_measure_community_detection["threshold_detection"]) + ".png")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        avgs[str(z)] = avg_list_ratio_orange
    # running the simulation (END)
    # plotting and saving the results (END)
    plt.clf()
    for i in range(1, 6):
        plt.plot(avgs[str(i)], label="orange nodes in " + twitter["name"] +
                                     " / threshold_detection = " +
                                     str(dict_counter_measure_community_detection["threshold_detection"]))
    plt.xlabel("rounds", fontdict=None, labelpad=None)
    plt.ylabel("the fraction of orange nodes", fontdict=None, labelpad=None)
    plt.title("Comparison of averages")
    plt.legend()
    plt.savefig("Output/average comparisons.png")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
