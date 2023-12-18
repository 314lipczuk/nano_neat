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
Problem: reader
I want to open an organism from a running experiment and see how it works
"""

def main():
    selected = False
    if len(sys.argv) == 2:
        file = sys.argv[1]
        selected = True
        realname = f"{file}.pkl"
        gn = Genome.create_from_file(realname)
        env = gym.make(ENVIRONMENT_NAME, render_mode='human')
        env = gym.make(ENVIRONMENT_NAME, render_mode='human')
        realname = f"{EXPERIMENT_NAME}/{NAME}/gen_{gen}.pkl"
        setattr(Genome, 'activation_function', activation_function)
        setattr(Genome, 'calculate_fitness', calculate_fitness)
        print(gn)
        Genome.calculate_fitness = calculate_fitness
        gn.calculate_fitness(self=gn, visible=True)
        #os.system(f"open {realname[:-3]}html")

    if len(sys.argv) == 3:
        gen= sys.argv[2]
        NAME = sys.argv[1]
        realname = f"./tmp/BIPEDAL/{NAME}/gen_{gen}.pkl"
        #setattr(Genome, 'activation_function', activation_function)
        #setattr(Genome, 'calculate_fitness', calculate_fitness)
        gn = Genome.create_from_file(realname)
        gn.config = DefaultWalkerConfig
        setattr(Genome, 'activation_function', gn.config.activation)
        setattr(Genome, 'calculate_fitness', gn.config.fitness)
        env = gym.make(gn.config.meta['gym_environment'], render_mode='human')
        gn.calculate_fitness(visible=True)
        os.system(f"open {realname[:-3]}html")

if __name__ == "__main__":
    main()