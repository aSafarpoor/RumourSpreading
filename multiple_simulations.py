from Simulation import *
import matplotlib.pyplot as plt
from Utility.dataset_setup import *
import sys


def community_detection_simulation(num_experiments, dataset, parameters_list, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    avg_outputs = list()
    avg_outputs_edges = list()
    for j in range(len(parameters_list)):
        outputs = [0] * num_experiments
        outputs_edges = [0] * num_experiments
        avg_output = list()
        avg_output_edges = list()
        max_len = 0
        for i in range(num_experiments):
            dict_counter_measure_community_detection = parameters_list[j]
            [list_num_white, list_num_red, list_num_orange, list_num_green, list_blocked_edges] = \
                simulation(realworld_graph=dataset, num_red=1, orange_p=0,
                           k=5, visualization=False, dict_args=None, dict_counter_measure=
                           dict_counter_measure_community_detection, seed=None)
            outputs[i] = list_num_orange
            outputs_edges[i] = list_blocked_edges
            if max_len < len(outputs[i]):
                max_len = len(outputs[i])
                avg_output = [0] * max_len
                avg_output_edges = [0] * max_len
        f = open(
            output_folder + '/output_' + str(dict_counter_measure_community_detection["threshold_detection"]) + ".txt",
            'w')
        for i in range(num_experiments):
            outputs[i].extend([outputs[i][-1]] * (max_len - len(outputs[i])))
            outputs_edges[i].extend([outputs_edges[i][-1]] * (max_len - len(outputs_edges[i])))
            f.write("outputs[ " + str(i) + " ]=" + repr(outputs[i]) + '\n')
            f.write("outputs_edges[ " + str(i) + " ]=" + repr(outputs_edges[i]) + '\n')
            f.write("--------------------------------------------\n")
            avg_output = [sum(x) for x in zip(avg_output, outputs[i])]
            avg_output_edges = [sum(x) for x in zip(avg_output_edges, outputs_edges[i])]

        avg_output = [x / num_experiments for x in avg_output]
        f.write("avg_output=" + repr(avg_output) + '\n')

        avg_output_edges = [x / num_experiments for x in avg_output_edges]
        f.write("avg_output_edges=" + repr(avg_output_edges) + '\n')
        f.close()

        plt.clf()
        plt.xlabel("rounds", fontdict=None, labelpad=None)
        plt.ylabel("the fraction of orange nodes", fontdict=None, labelpad=None)
        plt.title(
            "The average ratios of " + str(num_experiments) + " experiments on " + dataset["name"] + r" with $T_{n}=$"
            + str(dict_counter_measure_community_detection["threshold_detection"]))
        plt.plot(avg_output, label=dataset["name"] + " " + str(parameters_list[j]["threshold_detection"]), linewidth=2)

        leg = plt.legend()
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        plt.setp(leg_lines, linewidth=2)
        plt.setp(leg_texts, fontsize='large')
        plt.savefig(output_folder+'/' + str(dataset["name"]) + "-" + str(j))
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

        # plot edges
        plt.clf()
        plt.xlabel("rounds", fontdict=None, labelpad=None)
        plt.ylabel("the number of blocked", fontdict=None, labelpad=None)
        plt.title(
            "The average number of blocked edges in " + str(num_experiments) + " experiments on " + dataset[
                "name"] + r" with $T_{n}=$"
            + str(dict_counter_measure_community_detection["threshold_detection"]))
        plt.plot(avg_output_edges, label=dataset["name"] + " " + str(parameters_list[j]["threshold_detection"]),
                 linewidth=2)

        leg = plt.legend()
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        plt.setp(leg_lines, linewidth=2)
        plt.setp(leg_texts, fontsize='large')
        plt.savefig(output_folder+'/' + str(dataset["name"]) + "-" + str(j) + "-edges")
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        # plot edges

        avg_outputs.append(avg_output)
        avg_outputs_edges.append(avg_output_edges)

    plt.clf()
    plt.title(
        "The average ratios of " + str(num_experiments) + " experiments on " + dataset["name"] +
        r" with different $T_{n}$ values")
    plt.xlabel("rounds", fontdict=None, labelpad=None)
    plt.ylabel("the fraction of orange nodes", fontdict=None, labelpad=None)
    for j in range(len(parameters_list)):
        plt.plot(avg_outputs[j], label=dataset["name"] + r"-$T_{n}=$ " + str(parameters_list[j]["threshold_detection"]),
                 linewidth=2)
        leg = plt.legend()
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        plt.setp(leg_lines, linewidth=2)
        plt.setp(leg_texts, fontsize='large')
    plt.savefig(output_folder+'/' + str(dataset["name"]) + "-averages")
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

    # ****
    plt.clf()
    plt.title(
        "The average number of blocked edges in " + str(num_experiments) + " experiments on " + dataset["name"] +
        r" with different $T_{n}$ values")
    plt.xlabel("rounds", fontdict=None, labelpad=None)
    plt.ylabel("the number of blocked edges", fontdict=None, labelpad=None)
    for j in range(len(parameters_list)):
        plt.plot(avg_outputs_edges[j], label=dataset["name"] + r"-$T_{n}=$ "
                                             + str(parameters_list[j]["threshold_detection"]),
                 linewidth=2)
        leg = plt.legend()
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        plt.setp(leg_lines, linewidth=2)
        plt.setp(leg_texts, fontsize='large')
    plt.savefig(output_folder+'/' + str(dataset["name"]) + "-averages-edges")
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
    # ****


def green_info_simulation(num_experiments, dataset, parameters_list, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    avg_outputs = list()
    for j in range(len(parameters_list)):
        outputs = [0] * num_experiments
        avg_output = list()
        max_len = 0
        for i in range(num_experiments):
            dict_counter_measure_green_information = parameters_list[j]
            [list_num_white, list_num_red, list_num_orange, list_num_green,list_num_blocked_edges] = \
                simulation(realworld_graph=dataset, num_red=1, orange_p=0,
                           k=5, visualization=False, dict_args=None, dict_counter_measure=
                           dict_counter_measure_green_information, seed=None)
            outputs[i] = list_num_orange
            if max_len < len(outputs[i]):
                max_len = len(outputs[i])
                avg_output = [0] * max_len

        f = open(output_folder + '/output_' + str(dict_counter_measure_green_information["start_time"]) + ".txt", 'w')
        for i in range(num_experiments):
            outputs[i].extend([outputs[i][-1]] * (max_len - len(outputs[i])))
            f.write("outputs[ " + str(i) + " ]=" + repr(outputs[i]) + '\n')
            f.write("--------------------------------------------\n")
            avg_output = [sum(x) for x in zip(avg_output, outputs[i])]

        avg_output = [x / num_experiments for x in avg_output]
        f.write("avg_output=" + repr(avg_output) + '\n')
        f.close()

        plt.clf()
        plt.xlabel("rounds", fontdict=None, labelpad=None)
        plt.ylabel("the fraction of orange nodes", fontdict=None, labelpad=None)
        plt.title(
            "The average ratios of " + str(num_experiments) + " experiments on " + dataset["name"] + r" with $\Delta=$"
            + str(dict_counter_measure_green_information["start_time"]))
        plt.plot(avg_output, label=dataset["name"] + " " + str(parameters_list[j]["start_time"]), linewidth=2)

        leg = plt.legend()
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        plt.setp(leg_lines, linewidth=2)
        plt.setp(leg_texts, fontsize='large')
        plt.savefig(output_folder+'/' + str(dataset["name"]) + "-" + str(j))
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

        avg_outputs.append(avg_output)

    plt.clf()
    plt.title(
        "The average ratios of " + str(num_experiments) + " experiments on " + dataset["name"] +
        r" with different $\Delta$ values")
    plt.xlabel("rounds", fontdict=None, labelpad=None)
    plt.ylabel("the fraction of orange nodes", fontdict=None, labelpad=None)
    for j in range(len(parameters_list)):
        plt.plot(avg_outputs[j], label=dataset["name"] + r"-$\Delta=$ " + str(parameters_list[j]["start_time"]) +
                                       ", high degree: " + (
                                           "True" if parameters_list[j]["high_degree_selection_strategy"] else "False"),
                 linewidth=2)
        leg = plt.legend()
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        plt.setp(leg_lines, linewidth=2)
        plt.setp(leg_texts, fontsize='large')
    plt.savefig(output_folder+'/' + str(dataset["name"]) + "-averages")
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


def doubt_simulation(num_experiments, dataset, parameters_list, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    avg_outputs = list()
    for j in range(len(parameters_list)):
        outputs = [0] * num_experiments
        avg_output = list()
        max_len = 0
        for i in range(num_experiments):
            dict_counter_measure_doubt_spreading = parameters_list[j]
            [list_num_white, list_num_red, list_num_orange, list_num_green, list_num_blocked_edges] = \
                simulation(realworld_graph=dataset, num_red=1, orange_p=0,
                           k=5, visualization=False, dict_args=None, dict_counter_measure=
                           dict_counter_measure_doubt_spreading, seed=None)
            outputs[i] = list_num_orange
            if max_len < len(outputs[i]):
                max_len = len(outputs[i])
                avg_output = [0] * max_len
        f = open(output_folder + '/output_negative_doubt_shift_' +
                 str(dict_counter_measure_doubt_spreading["negative_doubt_shift"]) + "_positive_doubt_shift_" +
                 str(dict_counter_measure_doubt_spreading["positive_doubt_shift"]) + ".txt", 'w')
        for i in range(num_experiments):
            outputs[i].extend([outputs[i][-1]] * (max_len - len(outputs[i])))
            f.write("outputs[ " + str(i) + " ]=" + repr(outputs[i]) + '\n')
            f.write("--------------------------------------------\n")
            avg_output = [sum(x) for x in zip(avg_output, outputs[i])]

        avg_output = [x / num_experiments for x in avg_output]
        f.write("avg_output=" + repr(avg_output) + '\n')
        f.close()

        plt.clf()
        plt.xlabel("rounds", fontdict=None, labelpad=None)
        plt.ylabel("the fraction of orange nodes", fontdict=None, labelpad=None)
        plt.title(
            "The average ratios of " + str(num_experiments) + " experiments on " + dataset["name"] +
            " with pd = " + str(dict_counter_measure_doubt_spreading["positive_doubt_shift"]) +
            " with nd = " + str(dict_counter_measure_doubt_spreading["negative_doubt_shift"]))
        plt.plot(avg_output, label=dataset["name"] + "_pd=" + str(parameters_list[j]["positive_doubt_shift"])
                                   + "_nd=" + str(parameters_list[j]["negative_doubt_shift"]), linewidth=2)

        leg = plt.legend()
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        plt.setp(leg_lines, linewidth=2)
        plt.setp(leg_texts, fontsize='large')
        plt.savefig(output_folder+'/' + str(dataset["name"]) + "-" + str(j))
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

        avg_outputs.append(avg_output)

    plt.clf()
    plt.title(
        "The average ratios of " + str(num_experiments) + " experiments on " + dataset["name"] +
        r" with different pd and nd values")
    plt.xlabel("rounds", fontdict=None, labelpad=None)
    plt.ylabel("the fraction of orange nodes", fontdict=None, labelpad=None)
    for j in range(len(parameters_list)):
        plt.plot(avg_outputs[j],
                 label=dataset["name"] + "_pd=" + str(parameters_list[j]["positive_doubt_shift"]) + "_nd=" + str(
                     parameters_list[j]["negative_doubt_shift"]),
                 linewidth=2)
        leg = plt.legend()
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        plt.setp(leg_lines, linewidth=2)
        plt.setp(leg_texts, fontsize=12)
    plt.savefig(output_folder+'/' + str(dataset["name"]) + "-averages")
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


if __name__ == "__main__":
    dataset = []
    if sys.argv[1] == "0":
        dataset = facebook
    else:
        dataset = twitter

    parameters_list = list()
    if sys.argv[2] == "0":
        parameters_list.append({"type": COUNTER_MEASURE_COMMUNITY_DETECTION, "threshold_detection": 0.01,
                                "threshold_block": 0.05})
        parameters_list.append({"type": COUNTER_MEASURE_COMMUNITY_DETECTION, "threshold_detection": 0.05,
                                "threshold_block": 0.05})
        parameters_list.append({"type": COUNTER_MEASURE_COMMUNITY_DETECTION, "threshold_detection": 0.1,
                                "threshold_block": 0.05})
        parameters_list.append({"type": COUNTER_MEASURE_COMMUNITY_DETECTION, "threshold_detection": 0.15,
                                "threshold_block": 0.05})
        parameters_list.append({"type": COUNTER_MEASURE_COMMUNITY_DETECTION, "threshold_detection": 0.2,
                                "threshold_block": 0.05})
        community_detection_simulation(int(sys.argv[3]), dataset, parameters_list, sys.argv[4])
    elif sys.argv[2] == "1":
        parameters_list.append({"type": COUNTER_MEASURE_GREEN_INFORMATION, "start_time": 1,
                                "num_green": 1, "green_spreading_ratio": 0.5,
                                "high_degree_selection_strategy": True})
        parameters_list.append({"type": COUNTER_MEASURE_GREEN_INFORMATION, "start_time": 2,
                                "num_green": 1, "green_spreading_ratio": 0.5,
                                "high_degree_selection_strategy": True})
        parameters_list.append({"type": COUNTER_MEASURE_GREEN_INFORMATION, "start_time": 4,
                                "num_green": 1, "green_spreading_ratio": 0.5,
                                "high_degree_selection_strategy": True})
        parameters_list.append({"type": COUNTER_MEASURE_GREEN_INFORMATION, "start_time": 8,
                                "num_green": 1, "green_spreading_ratio": 0.5,
                                "high_degree_selection_strategy": True})
        parameters_list.append({"type": COUNTER_MEASURE_GREEN_INFORMATION, "start_time": 16,
                                "num_green": 1, "green_spreading_ratio": 0.5,
                                "high_degree_selection_strategy": True})

        green_info_simulation(int(sys.argv[3]), dataset, parameters_list, sys.argv[4])
    else:
        parameters_list.append({"type": COUNTER_MEASURE_DOUBT_SPREADING, "negative_doubt_shift": -0.05,
                                "positive_doubt_shift": 0.05})
        parameters_list.append({"type": COUNTER_MEASURE_DOUBT_SPREADING, "negative_doubt_shift": -0.1,
                                "positive_doubt_shift": 0.1})
        parameters_list.append({"type": COUNTER_MEASURE_DOUBT_SPREADING, "negative_doubt_shift": -0.15,
                                "positive_doubt_shift": 0.15})
        parameters_list.append({"type": COUNTER_MEASURE_DOUBT_SPREADING, "negative_doubt_shift": -0.15,
                                "positive_doubt_shift": 0.1})
        parameters_list.append({"type": COUNTER_MEASURE_DOUBT_SPREADING, "negative_doubt_shift": -0.1,
                                "positive_doubt_shift": 0.15})
        doubt_simulation(int(sys.argv[3]), dataset, parameters_list, sys.argv[4])
