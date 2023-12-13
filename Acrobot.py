from model import Population, Config, Genome , Specie, Initialize
import gymnasium as gym
import math

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
        environment = gym.make(ENVIRONMENT_NAME, render_mode='human')
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
        print(reward)
        fin_reward += reward
        if  terminated or truncated:
            done = True
    self.fitness = fin_reward

ENVIRONMENT_NAME = 'Acrobot-v1'
EXPERIMENT_NAME = './tmp/ACROBOT'
def display(self:Genome):
    environment = gym.make(ENVIRONMENT_NAME, render_mode='human')
    observation, info = environment.reset()
    done = False       
    # take in observations
    final_reward = 0
    environment.render()
    while not done:
        self.load_inputs(observation)
        self.run_network()
        output = self.get_output()
        assert(len(output) == 4)
        action = output
        observation, reward, terminated, truncated, info = environment.step(action)
        print('reward', reward)
        final_reward += reward
        if  terminated or truncated:
            done = True
            observation, info = environment.reset()
    print(f"Fitness of running organism: {final_reward}")
    

c = Config(
    activation=activation_function,
    display=display, 
    fitness=calculate_fitness,
    input_size=5,
    output_size=3,
    DYNAMIC_THRESHOLD=True

    )

Initialize(c)
p = Population(ENVIRONMENT_NAME=ENVIRONMENT_NAME, EXPERIMENT_NAME=EXPERIMENT_NAME)
Population.env = gym.make(ENVIRONMENT_NAME, render_mode='none')
p.run()
if p.done:
    new_env = gym.make(ENVIRONMENT_NAME, render_mode='human')
    p.champion.render_run(new_env)
    p.champion.show(name=f"{EXPERIMENT_NAME}/champion")
