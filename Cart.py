import torch
import copy
import math
import random
import networkx as nx
import gymnasium as gym
from pyvis.network import Network as VisualNetwork

"""
Problem: cartpole
i need to supply api for gymnasium - will have to change how evaluating a model works 
Todos:
- recurrent connections
- evaluating a model continuously
- a nice fit into gymnasium API 
- numpy numerical stuff

Okej, jak to zacząć?
Myślę że optymalnie będzie zacząć od ogarnięcia api do gymnasium,
potem zacząć modyfikować ewaluację coby pasowało, a na końcu odpalić rekurencyjne połączenia.
Wtedy jezeli będę musiał, to pododawać numpy'a
"""

class Node:
    next_node_id=0

    def __init__(self, node_id, layer=None, node_type='hidden'):
        self.node_id=node_id
        self.node_type=node_type
        self.node_layer= layer if layer is not None else 0 if node_type == 'input' else 0
        self.sum_input =0
        self.sum_output=0
    
    def __repr__(self):
        return f"{self.node_type} Node{self.node_id}, L{self.node_layer}"

    @staticmethod #working
    def get_node_id(genome):
        return len(genome.nodes)

class Connection:
    next_innov_id=0
    innov_table = {}
    def __init__(self, innov_id, in_node, out_node, weight=1.0, enabled=True, is_recurrent=False):
        self.innov_id=innov_id
        self.in_node=in_node
        self.out_node=out_node
        self.weight=weight
        self.enabled=enabled
        self.is_recurrent=is_recurrent

    def __hash__(self):
        return hash(self.innov_id)

    def __repr__(self):
        return f"Conn ${self.innov_id}, [{self.in_node} -> {self.out_node}] {'enabled' if self.enabled else 'disabled'}"
    
    @staticmethod
    def get_innov_id(connection_tuple:tuple[int,int]):
        exists = Connection.innov_table.get(connection_tuple)
        if exists is not None:
            return exists
        else:
            tmp = Connection.next_innov_id
            Connection.innov_table[connection_tuple] = tmp
            Connection.next_innov_id+=1
            return tmp

class Genome:
    C1 = 1
    C2 = 1
    C3 = 0.4
    chance_mutate_weight = 0.8
    chance_of_20p_weight_change = 0.9
    chance_of_new_random_weight = 0.1
    chance_to_add_connection = 0.05
    chance_to_add_node= 0.05
    tries_to_make_connections = 20
    chance_to_reactivate_disabled_connection = 0.25
    allowRecurrent = False

    def __init__(self, inputN, outputN):
        self.nodes = []
        self.connections = []
        self.fitness = 0
        for _ in range(inputN):
            self.nodes.append(Node(node_id=Node.get_node_id(self), node_type='input', layer=0))
        for _ in range(outputN):
            self.nodes.append(Node(node_id=Node.get_node_id(self), node_type='output', layer=1))
        
        for i in [i for i in self.nodes if i.node_type == 'input']:
            for o in [o for o in self.nodes if o.node_type == 'output']:
                self.connections.append(Connection(innov_id=Connection.get_innov_id((i.node_id, o.node_id)), in_node=i.node_id, out_node=o.node_id, weight=random.random()))
        self.refresh_layers()


    def mutate_add_node(self):
        conn_to_pick = random.choice([c for c in self.connections if c.enabled and c.is_recurrent == False])
        conn_to_pick.enabled = False
        
        new_node = Node(node_id=Node.get_node_id(self), layer=1, node_type='hidden')
        c1 = Connection(innov_id=Connection.get_innov_id((conn_to_pick.in_node, new_node.node_id)),\
                        in_node=conn_to_pick.in_node, out_node=new_node.node_id,\
                        weight=1.0, enabled=True, is_recurrent=False)
        c2 = Connection(innov_id=Connection.get_innov_id((new_node.node_id, conn_to_pick.out_node )),\
                        in_node=new_node.node_id, out_node=conn_to_pick.out_node,\
                        weight=conn_to_pick.weight, enabled=True, is_recurrent=False)
        self.nodes.append(new_node)
        self.connections.append(c1)
        self.connections.append(c2)
        self.refresh_layers()

    def mutate_add_connection(self):
        self.refresh_layers()
        counter = 0
        while counter < Genome.tries_to_make_connections:
            fr, to = random.choices(self.nodes, k=2)
            is_recurrent = fr.node_layer > to.node_layer
            existing = [c for c in self.connections if c.in_node == fr.node_id and c.out_node == to.node_id]
            if len(existing) != 0:
                if existing[0].enabled == False and random.random() < Genome.chance_to_reactivate_disabled_connection:
                    existing[0].enabled = True
                    return
                else:
                    counter += 1
                    continue

            if fr.node_layer == to.node_layer or\
                fr.node_id == to.node_id or\
                (Genome.allowRecurrent == False and is_recurrent) or\
                fr.node_type == 'output' or\
                to.node_type == 'input': 
                counter += 1
                continue
            
            conn = Connection(innov_id=Connection.get_innov_id((fr.node_id, to.node_id)),\
                in_node=fr.node_id, out_node=to.node_id, weight=random.uniform(-1,1), enabled=True, is_recurrent=is_recurrent)
            self.connections.append(conn)
            break

    def mutate_weights(self):
        for c in self.connections:
            if random.random() < Genome.chance_of_20p_weight_change:
                c.weight += c.weight * (0.2 if random.random() < 0.5 else -0.2) 
            else:
                c.weight = random.uniform(-1, 1)
    def mutate(self):
        r = random.random()
        if random.random() < Genome.chance_mutate_weight:
            self.mutate_weights()

        if random.random() < Genome.chance_to_add_node:
            self.mutate_add_node()
        
        if random.random() < Genome.chance_to_add_connection:
            self.mutate_add_connection()

    def activation_function(self, i):
        #x = i
        #return 1 / (1+math.exp(-x))

        #steepened sigmoid
        x = torch.tensor(i)
        if x >= 0:
            return float(1./(1+torch.exp(-1e5*x)).to(torch.float))
        else:
            return float(torch.exp(1e5*x)/(1+torch.exp(1e5*x)).to(torch.float))

    def __repr__(self):
        return f"Genome f:${self.fitness}, nodes:{len(self.nodes)}, conns:{len(self.connections)}"
    
    def refresh_layers(self):
        inputs = [n.node_id for n in self.nodes if n.node_type == 'input']
        def find_max_len_to_input(node_id):
            conns = [1 if c.in_node in inputs else 1+find_max_len_to_input(c.in_node) for c in self.connections if c.enabled and not c.is_recurrent and c.out_node == node_id]
            return max(conns, default=0)
            
        for n in [n for n in self.nodes if n.node_type != 'input']:
            n.node_layer = find_max_len_to_input(n.node_id)
        maxL = max([n.node_layer for n in self.nodes])
        for n in [n for n in self.nodes if n.node_type == 'output']:
            n.node_layer = maxL+1

    def show(self, hide_disabled=False):
        graph = nx.Graph()
        nt = VisualNetwork(notebook=True , cdn_resources='in_line',layout=True, directed=True)
        for n in self.nodes:
            nt.add_node(
                n.node_id,
                label=f"{n.node_type},{n.node_id}",
                level = n.node_layer
            )
        for c in self.connections:
            nt.add_edge(source=c.in_node, \
                        to=c.out_node, \
                        color='blue' if c.enabled and c.is_recurrent else 'red' if not c.enabled else 'black',\
                        title=f"in:{c.innov_id}\nw:{c.weight}",\
                        hidden=(hide_disabled and (not c.enabled)))
        nt.from_nx(graph) 
        nt.show('example.html')

    def load_inputs(self, inputs):
        for n, i in zip(list(sorted([n for n in self.nodes if n.node_type == 'input' ], key=lambda x: x.node_id)), inputs ):
            n.sum_output = i
            n.sum_input = i

    def run_network(self):
        layers = sorted(list(set([n.node_layer for n in self.nodes])))
        for l in layers[1:]:
            nodes = [n for n in self.nodes if n.node_layer == l]
            for n in nodes:
                conns = [c for c in self.connections if c.out_node == n.node_id and c.enabled]
                #n.sum_input = sum(c.weight * [nd for nd in self.nodes if c.in_node == nd.node_id][0].sum_output for c in conns)
                n.sum_input = 0
                for c in conns:
                    n.sum_input += c.weight * [nd for nd in self.nodes if c.in_node == nd.node_id][0].sum_output
                res =  self.activation_function(n.sum_input)
                n.sum_output = res

    def get_output(self):   
        return [n.sum_output for n in self.nodes if n.node_type == 'output']

    def compatibility_distance(self, other):
        n1, n2 = max([n.innov_id for n in self.connections if n.enabled]), max([n.innov_id for n in other.connections if n.enabled])
        N = max(n1,n2)
        excess_count = len([n for n in self.connections if n.innov_id > N and n.enabled]) + len([n for n in other.connections if n.innov_id > N and n.enabled])
        disjoint_count = len([ n for n in self.connections if n.innov_id <= N and n.enabled and n.innov_id not in [c.innov_id for c in other.connections if c.enabled]])\
            + len([ n for n in other.connections if n.innov_id <= N and n.enabled and n.innov_id not in [c.innov_id for c in self.connections if c.enabled]])

        common = [n.innov_id for n in self.connections if n.innov_id <= N and n.enabled and n.innov_id in [c.innov_id for c in other.connections if c.enabled]]
        a = [n.weight for n in sorted(self.connections, key=lambda x: x.innov_id) if n.innov_id in common]
        b = [n.weight for n in sorted(other.connections, key=lambda x: x.innov_id) if n.innov_id in common]

        weight_diff = sum([ abs(a[i]-b[i]) for i in range(len(a))]) / max(len(a), 1)
        return Genome.C1 * excess_count + Genome.C2 * disjoint_count + Genome.C3 * weight_diff
    
    def calculate_fitness(self, environment: gym.Env):
        observation,info = environment.reset()
        done = False       
        # take in observations
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
                #observation, info = environment.reset()

        self.fitness = fin_reward
        """
        results = []
        for i, o in zip(inputs, expected_outputs):
            self.load_inputs(i)
            self.run_network()
            output = self.get_output()[0]
            results.append((o, output ))
        error = sum( [abs( r - o ) for o, r in results] )
        self.fitness = round((4 - error) ** 2, 2)
        """

    def render_run(self, environment: gym.Env):
        observation, info = environment.reset()
        done = False       
        # take in observations
        final_reward = 0
        environment.render()
        while not done:
            self.load_inputs(observation)
            self.run_network()
            output = self.get_output()[0]
            action = 1 if output > 0.5 else 0
            observation, reward, terminated, truncated, info = environment.step(action)
            final_reward += reward
            if  terminated or truncated:
                done = True
                observation, info = environment.reset()
        print(f"Fitness of running organism: {final_reward}")

def crossover(genome1:Genome, genome2:Genome):
    def equals(g1,g2):
        equality_threshold = 0.015
        return abs(g1.fitness - g2.fitness) < equality_threshold

    def calculate_weight_for_common_genes(c1,c2, method='avg'):
        assert method == 'avg' or method == 'random'
        if method == 'avg': return (c1.weight + c2.weight) / 2 
        else: return c1.weight if random.random() < 0.5 else c2.weight 

    eq = equals(genome1, genome2)
    prototype = copy.deepcopy(
        (genome1 if random.random() < 0.5 else genome2) if eq else \
        genome1 if genome1.fitness > genome2.fitness else genome2
    )
    matching_genes = set(genome1.connections).intersection(set(genome2.connections))
    for con in [con for con in prototype.connections if con in matching_genes]:
        c1 = [c for c in genome1.connections if c.innov_id == con.innov_id][0]
        c2 = [c for c in genome2.connections if c.innov_id == con.innov_id][0]
        con.weight = calculate_weight_for_common_genes(c1,c2)
    
    prototype.mutate()
    return prototype


class Population:
    size = 50
    specie_target = 4
    max_iterations = 500
    threshold_step_size = 0.3
    problem_fitness_threshold = 100

    def __init__(self, size, genome, show=False):
        if show:
            self.env = gym.make('CartPole-v1', render_mode='human')
        else:
            self.env = gym.make('CartPole-v1', render_mode='none')
        self.organisms = []
        self.species = []
        self.generation = 0
        self.done = False
        Population.size = size
        Population.threshold_step_size = 0.3
        g = Genome(genome[0], genome[1])
        self.default_genome = g
        for _ in range(size+1):
            self.organisms.append(copy.deepcopy(g))
        self.speciate()

    def calculate_fitness(self):
        for o in self.organisms:
            o.calculate_fitness(self.env)

    def iteration(self):
        print("Generation", self.generation, "Best fitness", self.organisms[0].fitness)
        self.speciate()
        self.calculate_fitness()
        self.organisms.sort(key=lambda x: x.fitness, reverse=True)
        if self.organisms[0].fitness > Population.problem_fitness_threshold:
            self.done = True
            self.champion = self.organisms[0]
            return
        self.calculate_offspring()
        self.crossover()

    def run(self):
        while self.done == False and self.generation < Population.max_iterations:
            self.iteration()
            self.generation += 1
        if self.done:
            print("Done")
            #print("Champion", self.champion.fitness)
            #self.champion.show()
        else:
            print("Did not find solution in {} iterations".format(Population.max_iterations))
            #print(f"Species:, {len(self.species)}")

    def calculate_offspring(self):
        for s in self.species:
            s.calculate_stats()
        total_average = sum([s.average for s in self.species])
        for s in self.species:
            #s.offspring = math.floor((s.average / total_average) * s.n) #Population.size
            s.offspring = math.floor((s.average / total_average) * Population.size) #Population.size
            if s.gens_since_improvement >= 15:
                s.offspring = 0
            #print(f"Specie {s.id}, pop:${len(s.organisms)}, avg f{round(s.average, 3)} has {s.offspring} offspring when total av is ${round(total_average, 3)}")

    def crossover(self):
        new_population = []
        counter = 0

        OUT_OF_SPECIE_MODIFIER = 0.3
        if True: #TODO add check for elitism
            # top 5% of population goes to next generation
            best = sorted(self.organisms, key=lambda x:x.fitness, reverse=True)[:math.floor(Population.size * 0.05)] 
            for b in best: new_population.append(copy.deepcopy(b))

        for s in self.species:
            specie_counter = 0

            choices = []
            weights = []
            for x in [x for x in sorted(s.organisms, key=lambda f:f.fitness, reverse=True)[:max(math.floor(0.2 * len(s.organisms)), 1)] ]:
                choices.append(x)
                weights.append(x.fitness)
            #for x in [x for x in sorted(self.organisms, key=lambda f:f.fitness, reverse=True)[math.floor(0.2 * len(s.organisms) ):] ]:
            #    choices.append(x)
            #    weights.append(x.fitness * OUT_OF_SPECIE_MODIFIER)

            while specie_counter < s.offspring:
                g1, g2 = random.choices(choices, weights=weights, k=2)
                new_population.append(crossover(g1,g2))
                specie_counter+=1
        while len(new_population) < Population.size:
            counter += 1
            new_population.append(copy.deepcopy(self.default_genome))
        #print(f"generation {self.generation} had {counter} zeroed genomes")
        self.organisms = new_population

    def speciate(self):
        for s in self.species:
            s.clear_members()

        for g in self.organisms:
            added = False
            for s in self.species:
                cd = g.compatibility_distance(s.representative)
                if cd < Specie.threshold:
                    s.add_organism(g)
                    added = True
                    break
            if not added:
                self.species.append(Specie(id=len(self.species), representative=g))
        self.species = [spec for spec in self.species if len(spec.organisms) > 0]
        for s in self.species: 
            s.representative = random.choice(s.organisms)

        #setting new threshold to adjust specie size
        #specie_size = len(self.species)
        #new_threshold = Specie.threshold if specie_size == Population.specie_target \
        #    else Specie.threshold - Population.threshold_step_size if specie_size < Population.specie_target\
        #    else Specie.threshold + Population.threshold_step_size
        #Specie.threshold = new_threshold

class Specie:
    threshold = 3
    use_adjusted_fitness=True
    def __init__(self, id, representative):
        self.id = id
        self.organisms = []
        self.representative =  representative
        self.gens_since_improvement = 0
        self.average = 0

    def add_organism(self, organism):
        self.organisms.append(organism)

    def calculate_adjusted_fitness(self):
        N = len(self.organisms)
        for o in self.organisms:
            o.adjusted_fitness = o.fitness / N

    def clear_members(self):
        self.organisms = []
    
    def calculate_stats(self):
        oldAvg = self.average
        if Specie.use_adjusted_fitness:
            self.calculate_adjusted_fitness()
            self.average = sum([o.adjusted_fitness for o in self.organisms]) / len(self.organisms)
        else:
            self.average = sum([o.fitness for o in self.organisms]) / len(self.organisms)
        if self.average >= oldAvg:
            self.gens_since_improvement = 0
        else:
            self.gens_since_improvement += 1
        self.n = len(self.organisms)

    def __repr__(self):
        return f"Specie id:{self.id}, len:{len(self.organisms)}, avg f{round(self.average, 3)}, gens since imp {self.gens_since_improvement}, avg nodes {sum([len(o.nodes) for o in self.organisms]) / len(self.organisms)}, avg conns: {sum([len(o.connections) for o in self.organisms]) / len(self.organisms)}"

def main():
    p = Population(50, (4,1), show=False)
    p.run()
    if p.done:
        new_env = gym.make('CartPole-v1', render_mode='human')
        p.champion.render_run(new_env)
        p.champion.show()

main()