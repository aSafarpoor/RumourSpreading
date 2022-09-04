import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from IMP import twitter_loc
from IMP.Simulation import Simulation

if __name__ == "__main__":
    Simulation(graph="whatev", SNtype=False, type_graph="KClique", p=1/50000, gray_p=0, k=5, c=10000, tresh=0.10,
               d=0, j=5)