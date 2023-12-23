from model import Population, Config, Genome , Specie
import gymnasium as gym
import math
import dill
import pickle
import torch

def activation_function(self, x):
    x = torch.tensor(x)
    if x >= 0:
        return float(1./(1+torch.exp(-1e5*x)).to(torch.float))
    else:
        return float(torch.exp(1e5*x)/(1+torch.exp(1e5*x)).to(torch.float))

def calculate_fitness(self, visible=False):
    INPUT = [[0,0], [0,1], [1,0], [1,1]]
    OUTPUT = [0, 1, 1, 0]
    results = []
    for i, o in zip(INPUT,OUTPUT):
        self.load_inputs(i)
        self.run_network()
        output = self.get_output()[0]
        results.append((o, output ))
    error = sum( [abs( r - o ) for o, r in results] )
    fitness = round((4 - error) ** 2, 2)
    isDone = fitness > 15.5
    return (isDone, fitness)

def every_generation(p:Population):
    if p.generation % 10 == 0 and p.generation > 0:
            name = f"gen_{p.generation}"
            f = open(f"{p.path}/{name}.pkl", "wb")
            dill.dump(p.organisms[0], f )
            f.close()
            print(f"Saved model {p.serial_number} {name}")

def after_finished(p:Population):
    champ = p.champion
    #champ.calculate_fitness(visible=True)
    champStats = f"{p.generation} {len(champ.nodes)} {len(champ.connections)} "
    with open(f"{p.path}/champStats.txt", "w") as f:
        f.write(champStats)

XorConfig = Config(
    activation=activation_function,
    fitness=calculate_fitness, 
    input_size=2, 
    EXPERIMENT_PATH= './tmp/XOR',
    output_size=1 ,
    meta = {},
    every_generation=every_generation,
    after_finished=after_finished,
    max_iterations=300,
    )

def main():
    p = Population(config=XorConfig)
    p.run()

if __name__ == "__main__":
    main()