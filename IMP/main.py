import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from IMP import twitter_loc
from IMP.simulation import Simulation

if __name__ == "__main__":
    Simulation(graph="whatev", SNtype=False, type_graph="KCliqueExpander", p=0.1, gray_p=0.1, k=4, c=20, tresh=0.10,
               d=1, j=4)