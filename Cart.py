####
from model import Population, Config, Genome , Specie
import gymnasium as gym
import math
import dill
import pickle

def activation_function(self, x):
    return 1 / (1+math.exp(-x))

def calculate_fitness(self, visible=False):
    if visible:
        experiment_name = self.config.meta["gym_environment"]
        environment = gym.make(experiment_name, render_mode='human')
    else:
        environment = Population.env
    observation,info = environment.reset()
    done = False       
    fin_reward = 0
    while not done:
        self.load_inputs(observation)
        self.run_network()
        output = self.get_output()[0]
        action = 1 if output > 0.5 else 0
        observation, reward, terminated, truncated, info = environment.step(action)
        fin_reward += reward
        if  terminated or truncated:
            done = True
    self.fitness = fin_reward
    return (False, fin_reward)

def every_generation(p:Population):
    if p.generation % 10 == 0 and p.generation > 0:
            name = f"gen_{p.generation}"
            chosen =p.organisms[0]
            path = f"{p.path}/{name}"
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

DefaultCartConfig = Config(
    activation=activation_function,
    fitness=calculate_fitness, 
    input_size=4, 
    EXPERIMENT_PATH= './tmp/CART',
    output_size=1 ,
    meta = {"gym_environment": "CartPole-v1", "experiment_name": "CART" },
    every_generation= every_generation,
    after_finished= after_finished,
    problem_fitness_threshold= 499,
    )

def main():
    p = Population(config=DefaultCartConfig)
    experiment_name =DefaultCartConfig.meta["gym_environment"]
    Population.env = gym.make(experiment_name, render_mode='human')
    p.run()

if __name__ == "__main__":
    main()