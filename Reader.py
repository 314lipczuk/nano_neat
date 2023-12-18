import os
import copy
import math
import random
import networkx as nx
import gymnasium as gym
from pyvis.network import Network as VisualNetwork
import dill
import sys
from walker import DefaultWalkerConfig
from model import Genome

"""
Reader
    args: population_name generation_number
    runs simulation of the walker for saved generation of population specified
    I could do this better & more generic, but it's not worth it. 
    Other problems (acrobot & cart) are solved too quickly to be worth looking at.
"""

def main():
    if len(sys.argv) == 3:
        gen= sys.argv[2]
        NAME = sys.argv[1]
        realname = f"./tmp/BIPEDAL/{NAME}/gen_{gen}.pkl"
        gn = Genome.create_from_file(realname)
        gn.config = DefaultWalkerConfig
        setattr(Genome, 'activation_function', gn.config.activation)
        setattr(Genome, 'calculate_fitness', gn.config.fitness)
        env = gym.make(gn.config.meta['gym_environment'], render_mode='human')
        gn.calculate_fitness(visible=True)
        os.system(f"open {realname[:-3]}html")

if __name__ == "__main__":
    main()