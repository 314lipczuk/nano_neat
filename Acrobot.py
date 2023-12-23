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
        output = self.get_output()
        assert(len(output) == 3)
        action =  max(enumerate(output), key=lambda x : x[1])[0]
        observation, reward, terminated, truncated, info = environment.step(action)
        fin_reward += reward
        if terminated or truncated:
            done = True
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
    # generation n_nodes n_connections
    champStats = f"{p.generation} {len(champ.nodes)} {len(champ.connections)} "
    with open(f"{p.path}/champStats.txt", "w") as f:
        f.write(champStats)

DefaultAcrobotConfig = Config(
    activation=activation_function,
    fitness=calculate_fitness, 
    input_size=5, 
    EXPERIMENT_PATH= './tmp/ACROBOT',
    output_size=3 ,
    meta = {"gym_environment": "Acrobot-v1" },
    every_generation=every_generation,
    after_finished=after_finished,
    problem_fitness_threshold=-100,
    )

def main():
    p = Population(config=DefaultAcrobotConfig )
    experiment_name = DefaultAcrobotConfig.meta["gym_environment"]
    Population.env = gym.make(experiment_name, render_mode='none')
    p.run()
if __name__ == "__main__":
    main()