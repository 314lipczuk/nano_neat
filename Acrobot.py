#from model import Population, Config, Genome , Specie, Initialize
#import gymnasium as gym
#import math

#def activation_function(self, x):
#    try:
#        value =  2/(1+math.exp(-5*x))-1
#    except OverflowError:
#        if x > 1:
#            value = 1
#        if x < -1:
#            value = -1
#    return value
#
#def calculate_fitness(self, visible=False):
#    if visible:
#        environment = gym.make(ENVIRONMENT_NAME, render_mode='human')
#    else:
#        environment = Population.env
#    observation,info = environment.reset()
#    done = False       
#    fin_reward = 0
#    while not done:
#        self.load_inputs(observation)
#        self.run_network()
#        output = self.get_output()
#        assert(len(output) == 3)
#        action =  max(enumerate(output), key=lambda x : x[1])[0]
#        observation, reward, terminated, truncated, info = environment.step(action)
#        print(reward)
#        fin_reward += reward
#        if  terminated or truncated:
#            done = True
#    self.fitness = fin_reward
#
#ENVIRONMENT_NAME = 'Acrobot-v1'
#EXPERIMENT_NAME = './tmp/ACROBOT'
#def display(self:Genome):
#    environment = gym.make(ENVIRONMENT_NAME, render_mode='human')
#    observation, info = environment.reset()
#    done = False       
#    # take in observations
#    final_reward = 0
#    environment.render()
#    while not done:
#        self.load_inputs(observation)
#        self.run_network()
#        output = self.get_output()
#        assert(len(output) == 4)
#        action = output
#        observation, reward, terminated, truncated, info = environment.step(action)
#        print('reward', reward)
#        final_reward += reward
#        if  terminated or truncated:
#            done = True
#            observation, info = environment.reset()
#    print(f"Fitness of running organism: {final_reward}")
#    
#
#c = Config(
#    activation=activation_function,
#    display=display, 
#    fitness=calculate_fitness,
#    input_size=5,
#    output_size=3,
#    DYNAMIC_THRESHOLD=True
#    )
#
#Initialize(c)
#p = Population(ENVIRONMENT_NAME=ENVIRONMENT_NAME, EXPERIMENT_NAME=EXPERIMENT_NAME)
#Population.env = gym.make(ENVIRONMENT_NAME, render_mode='none')
#p.run()
#if p.done:
#    new_env = gym.make(ENVIRONMENT_NAME, render_mode='human')
#    p.champion.render_run(new_env)
#    p.champion.show(name=f"{EXPERIMENT_NAME}/champion")





from model import Population, Config, Genome , Specie
import gymnasium as gym
import math
import dill
import pickle

def activation_function(self, x):
    return 1 / (1+math.exp(-x))
    #try:
    #    value =  2/(1+math.exp(-5*x))-1
    #except OverflowError:
    #    if x > 1:
    #        value = 1
    #    if x < -1:
    #        value = -1
    #return value


#def calculate_fitness(self, visible=False):
#    if visible:
#        environment = gym.make(ENVIRONMENT_NAME, render_mode='human')
#    else:
#        environment = Population.env
#    observation,info = environment.reset()
#    done = False       
#    fin_reward = 0
#    while not done:
#        self.load_inputs(observation)
#        self.run_network()
#        output = self.get_output()
#        assert(len(output) == 3)
#        action =  max(enumerate(output), key=lambda x : x[1])[0]
#        observation, reward, terminated, truncated, info = environment.step(action)
#        print(reward)
#        fin_reward += reward
#        if  terminated or truncated:
#            done = True
#    self.fitness = fin_reward

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
        #if reward < 0.01:
        #    not_moving_counter += 1
        #    if not_moving_counter > 30:
        #        done = True
        #else:
        #    not_moving_counter = 0
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
    champ.calculate_fitness(visible=True)

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
    #if p.done:
    #    new_env = gym.make(experiment_name, render_mode='human')
    #    p.champion.render_run(new_env)
    #    p.champion.show(name=f"{experiment_name}/champion")

if __name__ == "__main__":
    main()