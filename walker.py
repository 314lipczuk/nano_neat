from model import Population, Config, Genome , Specie
import time
import gymnasium as gym
import math
import dill
import pickle

def activation_function(self, x):
    try:
        value =  2/(1+math.exp(-5*x))-1
    except OverflowError:
        if x > 1:
            value = 1
        if x < -1:
            value = -1
    return value

def calculate_fitness(self, visible=False):
    if visible:
        experiment_name = self.config.meta["gym_environment"]
        environment = gym.make(experiment_name, render_mode='human')
    else:
        environment = Population.env
    observation,info = environment.reset()
    done = False       
    fin_reward = 0
    not_moving_counter = 0
    while not done:
        self.load_inputs(observation)
        self.run_network()
        output = self.get_output()
        assert(len(output) == 4)
        action = output
        observation, reward, terminated, truncated, info = environment.step(action)
        if reward < 0.01:
            not_moving_counter += 1
            if not_moving_counter > 30:
                done = True
        else:
            not_moving_counter = 0
        fin_reward += reward
        if terminated or truncated:
            done = True
    fin_reward = max(fin_reward, -100) + 101
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
    champ.calculate_fitness(visible=True)

def print_generation(p:Population):
    mean = sum([o.fitness for o in p.organisms]) / len(p.organisms)
    std_dev = math.sqrt(sum([(o.fitness - mean) ** 2 for o in p.organisms]) / len(p.organisms))
    n = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    itertime = (n- p.t)/1000000000
    p.t = n
    print("Generation", p.generation, "\tfitness:", "%.2f" %(p.organisms[0].fitness - 100), "\tmean:", "%.2f" % (mean - 100), "\tstd dev:", "%.2f" %std_dev, "\tSpecies:", len(p.species), "Population ",len(p.organisms), "itertime:" ,round(itertime, 2), "s")

DefaultWalkerConfig = Config(
    activation=activation_function,
    fitness=calculate_fitness, 
    input_size=28, 
    EXPERIMENT_PATH= './tmp/BIPEDAL',
    output_size=4 ,
    meta = {"gym_environment": "BipedalWalker-v3", "experiment_name": "BIPEDAL" },
    every_generation=every_generation,
    after_finished=after_finished,
    DYNAMIC_THRESHOLD = True,
    problem_fitness_threshold = 390,
    CROSS_SPECIE_REPRODUCTION =True,
    print_generation = print_generation,
    )

def main():
    p = Population(config=DefaultWalkerConfig)
    experiment_name = DefaultWalkerConfig.meta["gym_environment"]
    Population.env = gym.make(experiment_name, render_mode='none')
    p.run()

if __name__ == "__main__":
    main()