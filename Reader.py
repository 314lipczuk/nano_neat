import os
import copy
import math
import random
import networkx as nx
import gymnasium as gym
from pyvis.network import Network as VisualNetwork
import pickle
import sys
from problem_example import calculate_fitness, activation_function 
from model import Genome

"""
Problem: reader
I want to open an organism from a running experiment and see how it works
"""

ENVIRONMENT_NAME = 'BipedalWalker-v3'
EXPERIMENT_NAME = './tmp/BIPEDAL'

ENVIRONMENT_NAME = 'Acrobot-v1'
EXPERIMENT_NAME = './tmp/ACROBOT'
def main():
    global ENVIRONMENT_NAME
    global EXPERIMENT_NAME
    selected = False
    if len(sys.argv) == 2:
        file = sys.argv[1]
        selected = True
        env = gym.make(ENVIRONMENT_NAME, render_mode='human')
        realname = f"{file}.pkl"
        gn = Genome.create_from_file(realname)
        #if gn.EXPERIMENT_NAME: 
        #    EXPERIMENT_NAME = gn.EXPERIMENT_NAME
        #if gn.ENVIRONMENT_NAME :
        #     ENVIRONMENT_NAME= gn.ENVIRONMENT_NAME
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
        env = gym.make(ENVIRONMENT_NAME, render_mode='human')
        realname = f"{EXPERIMENT_NAME}/{NAME}/gen_{gen}.pkl"
        setattr(Genome, 'activation_function', activation_function)
        setattr(Genome, 'calculate_fitness', calculate_fitness)
        gn = Genome.create_from_file(realname)
        gn.calculate_fitness(visible=True)
        os.system(f"open {realname[:-3]}html")
main()