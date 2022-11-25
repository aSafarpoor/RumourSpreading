from Simulation import *
import matplotlib.pyplot as plt
from Utility.dataset_setup import *


def community_detection_simulation(num_experiments, dataset, parameters_list):
    avg_outputs = list()
    for j in range(len(parameters_list)):
        outputs = [0] * num_experiments
        avg_output = list()
        max_len = 0
        for i in range(num_experiments):
            dict_counter_measure_community_detection = parameters_list[j]
            [list_num_white, list_num_red, list_num_orange, list_num_green] = \
                simulation(realworld_graph=dataset, num_red=1, orange_p=0,
                           k=5, visualization=False, dict_args=None, dict_counter_measure=
                           dict_counter_measure_community_detection, seed=None)
            outputs[i] = list_num_orange
            if max_len < len(outputs[i]):
                max_len = len(outputs[i])
                avg_output = [0] * max_len
        f = open('Output/output_'+str(dict_counter_measure_community_detection["threshold_detection"])+".txt", 'w')
        for i in range(num_experiments):
            outputs[i].extend([outputs[i][-1]] * (max_len - len(outputs[i])))
            f.write("outputs[ "+str(i)+" ]=" + repr(outputs[i]) + '\n')
            f.write("--------------------------------------------\n")
            avg_output = [sum(x) for x in zip(avg_output, outputs[i])]

        avg_output = [x / num_experiments for x in avg_output]
        f.write("avg_output=" + repr(avg_output) + '\n')
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
        plt.savefig('Output/' + str(dataset["name"]) + "-" + str(j))
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

        avg_outputs.append(avg_output)

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
    plt.savefig('Output/' + str(dataset["name"]) + "-averages")
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


if __name__ == "__main__":
    dataset = twitter
    parameters_list = list()
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
    community_detection_simulation(20, dataset, parameters_list)
