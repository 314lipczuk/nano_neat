import torch
import time
import dill
import os
import string
import copy
import math
import random
import networkx as nx
import gymnasium as gym
from pyvis.network import Network as VisualNetwork
import pickle

"""
Modification of the base model that replaces basic summation with Kalman filter
"""

raise NotImplementedError("This is not finished yet")

class Config:
    def __init__(self,
     activation
    ,fitness
    ,input_size
    ,output_size
    ,EXPERIMENT_PATH
    ,initialization_function = lambda x : x
    ,C1=1,
    C2=1,
    C3=0.4,
    meta={},
    print_generation = None,       
    every_generation = lambda x : x,
    after_finished = lambda x : x
    ,chance_mutate_weight = 0.8
    ,chance_of_20p_weight_change = 0.9
    ,chance_of_new_random_weight = 0.1
    ,chance_to_add_connection = 0.05
    ,chance_to_add_node= 0.05
    ,tries_to_make_connections = 20
    ,chance_to_reactivate_disabled_connection = 0.25
    ,allowRecurrent = True
    ,size = 50
    ,specie_target = 4
    ,max_iterations = 600
    ,threshold_step_size = 0.3
    ,problem_fitness_threshold = 390
    ,TOP_PROC_TO_REPRODUCE = 0.2
    ,CROSS_SPECIE_REPRODUCTION = True
    ,CROSS_SPECIE_WEIGHT_MODIFIER = 0.3
    ,ELITISM = True
    ,ELITISM_PERCENTAGE = 0.05
    ,DYNAMIC_THRESHOLD = True
    ,INITIAL_THRESHOLD = 3
     ):
        assert EXPERIMENT_PATH is not None
        assert activation is not None
        assert fitness is not None
        assert input_size is not None
        assert output_size is not None


        self.print_generation  = print_generation        
        self.every_generation =every_generation 
        self.after_finished = after_finished 

        self.activation = activation
        self.EXPERIMENT_PATH= EXPERIMENT_PATH
        self.initialization_function = initialization_function
        self.fitness = fitness
        self.input_size = input_size
        self.output_size = output_size
        self.meta = meta
        self.C1=C1
        self.C2=C2
        self.C3=C3
        self.chance_mutate_weight = chance_mutate_weight
        self.chance_of_20p_weight_change = chance_of_20p_weight_change 
        self.chance_of_new_random_weight = chance_of_new_random_weight 
        self.chance_to_add_connection = chance_to_add_connection 
        self.chance_to_add_node = chance_to_add_node 
        self.tries_to_make_connections = tries_to_make_connections 
        self.chance_to_reactivate_disabled_connection = chance_to_reactivate_disabled_connection 
        self.allowRecurrent = allowRecurrent 
        self.size = size
        self.specie_target = specie_target 
        self.max_iterations = max_iterations
        self.threshold_step_size =threshold_step_size 
        self.problem_fitness_threshold = problem_fitness_threshold 
        self.TOP_PROC_TO_REPRODUCE = TOP_PROC_TO_REPRODUCE 
        self.CROSS_SPECIE_REPRODUCTION = CROSS_SPECIE_REPRODUCTION 
        self.CROSS_SPECIE_WEIGHT_MODIFIER = CROSS_SPECIE_WEIGHT_MODIFIER 
        self.ELITISM = ELITISM
        self.ELITISM_PERCENTAGE = ELITISM_PERCENTAGE 
        self.DYNAMIC_THRESHOLD = DYNAMIC_THRESHOLD 
        self.INITIAL_THRESHOLD = INITIAL_THRESHOLD 

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

    @staticmethod 
    def get_node_id(genome):
        return len(genome.nodes)

class Connection:
    next_innov_id=0
    innov_table = {}
    def __init__(self, innov_id, in_node, out_node, weight=1.0, enabled=True, is_recurrent=False):
        self.innov_id = innov_id
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.is_recurrent = is_recurrent

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
    def __init__(self, config):
        self.config = config
        inputN, outputN = config.input_size, config.output_size
        self.nodes = []
        self.connections = []
        self.fitness = 0.01
        for _ in range(inputN):
            self.nodes.append(Node(node_id=Node.get_node_id(self), node_type='input', layer=0))
        for _ in range(outputN):
            self.nodes.append(Node(node_id=Node.get_node_id(self), node_type='output', layer=1))
        
        for i in [i for i in self.nodes if i.node_type == 'input']:
            for o in [o for o in self.nodes if o.node_type == 'output']:
                self.connections.append(Connection(innov_id=Connection.get_innov_id((i.node_id, o.node_id)), in_node=i.node_id, out_node=o.node_id, weight=random.uniform(-1,1)))
        self.refresh_layers()


    @staticmethod
    def create_from_file(path):
        fl = open(path, 'rb')
        f = dill.load(fl)
        fl.close()
        return f
    
    def save(self):
        name = f"gen_{self.generation}"
        path = f"{self.path}/{name}"
        pickle.dump(self, open(f"{self.path}/{name}.pkl", "wb"))
        print('saved')

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
        while counter < self.config.tries_to_make_connections:
            fr, to = random.choices(self.nodes, k=2)
            is_recurrent = fr.node_layer > to.node_layer
            existing = [c for c in self.connections if c.in_node == fr.node_id and c.out_node == to.node_id]
            if len(existing) != 0:
                if existing[0].enabled == False and random.random() < self.config.chance_to_reactivate_disabled_connection and fr.node_layer != to.node_layer:
                    existing[0].enabled = True
                    return
                else:
                    counter += 1
                    continue

            if fr.node_layer == to.node_layer or\
                fr.node_id == to.node_id or\
                (self.config.allowRecurrent == False and is_recurrent) or\
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
            if random.random() < self.config.chance_of_20p_weight_change:
                c.weight += c.weight * (0.2 if random.random() < 0.5 else -0.2) 
            else:
                c.weight = random.uniform(-1, 1)
    def mutate(self):
        r = random.random()
        if random.random() < self.config.chance_mutate_weight:
            self.mutate_weights()

        if random.random() < self.config.chance_to_add_node:
            self.mutate_add_node()
        
        if random.random() < self.config.chance_to_add_connection:
            self.mutate_add_connection()

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

        recurrent = [c for c in self.connections if c.is_recurrent]
        for r in recurrent:
            in_node = [n for n in self.nodes if n.node_id == r.in_node][0]
            out_node = [n for n in self.nodes if n.node_id == r.out_node][0]
            if in_node.node_layer == out_node.node_layer:
                r.enabled = False
            elif in_node.node_layer < out_node.node_layer:
                r.is_recurrent = False

    def save_brain(self, hide_disabled=False, name='example'):
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
        nt.show(f'{self.EXPERIMENT_PATH}/{name}.html')

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
                n.sum_input = 0
                """
                Here be kalman.  
                But, questions:
                - What is the model generating the predictions?
                - What are we trying to estimate? 
                    The inputs are not noisy, and if they were we should deal with that on input itself (?)



                """
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
        return self.config.C1 * excess_count + self.config.C2 * disjoint_count + self.config.C3 * weight_diff
    
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
    def __init__(self, config):
        self.config = config
        self.config = config
        setattr(Genome, 'calculate_fitness', config.fitness)
        setattr(Genome, 'activation_function', config.activation)
        self.organisms = []
        Specie.threshold = config.INITIAL_THRESHOLD
        self.species = []
        self.generation = 0
        self.total_adjusted_fitness = 0
        self.serial_number = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        self.t = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        self.path = f"{config.EXPERIMENT_PATH}/{self.serial_number}"
        os.mkdir(self.path)
        print(f"Starting experiment {self.serial_number}")
        self.done = False
        g = Genome(config)
        self.default_genome = g
        for _ in range(config.size+1):
            self.organisms.append(copy.deepcopy(g))
        self.speciate()

    def iteration(self):
        self.speciate()
        for o in self.organisms:
            isDone, fitness = o.calculate_fitness()
            o.fitness = fitness
            if isDone or o.fitness > self.config.problem_fitness_threshold:
                self.done = True
                self.champion = o
                self.config.after_finished(self)
                return True
        self.organisms.sort(key=lambda x: x.fitness, reverse=True)
        if self.config.print_generation is not None:
            self.config.print_generation(self)
        else:
            mean = sum([o.fitness for o in self.organisms]) / len(self.organisms)
            std_dev = math.sqrt(sum([(o.fitness - mean) ** 2 for o in self.organisms]) / len(self.organisms))
            n = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            itertime = (n-self.t)/1000000000
            self.t = n
            print("Generation", self.generation, "\tfitness:", "%.2f" %(self.organisms[0].fitness), "\tmean:", "%.2f" %mean, "\tstd dev:", "%.2f" %std_dev, "\tSpecies:", len(self.species), "Population ",len(self.organisms), "itertime:" ,round(itertime, 2), "s")
        if self.organisms[0].fitness > self.config.problem_fitness_threshold:
            self.done = True
            self.champion = self.organisms[0]
            self.config.after_finished(self)
            return True
        self.config.every_generation(self)
        self.calculate_offspring()
        self.crossover()

    def run(self):
        while self.done == False and self.generation < self.config.max_iterations:
            r = self.iteration()
            if r: break
            self.generation += 1
        if self.done:
            print(f"Experiment {self.serial_number} finished in {self.generation} generations")
        else:
            print(f"Experiment {self.serial_number} did not finish in {self.generation} generations")

    def calculate_offspring(self):
        for s in self.species:
            s.calculate_stats()
        total_average = sum([o.adjusted_fitness for o in self.organisms])
        for s in self.species:
            proportion = s.total_adjusted_fitness / total_average 
            s.offspring = round(proportion * self.config.size)
            if s.offspring > self.config.size:
                print("######## ABNORMAL SIZE OF SPECIE", s," #########\nproportion", proportion, " total: ", total_average, " specie_total", s.total_adjusted_fitness, " specie offspring", s.offspring, " population size", self.config.size)
            if s.gens_since_improvement >= 15:
                s.offspring = 0

    def crossover(self):
        new_population = []
        counter = 0

        if self.config.ELITISM:
            best = sorted(self.organisms, key=lambda x:x.fitness, reverse=True)[:math.floor( self.config.size * self.config.ELITISM_PERCENTAGE)] 
            for b in best: new_population.append(copy.deepcopy(b))

        for s in self.species:
            specie_counter = 0
            choices = []
            weights = []
            for x in [x for x in sorted(s.organisms, key=lambda f:f.fitness, reverse=True)[: max(math.floor( self.config.TOP_PROC_TO_REPRODUCE * len(s.organisms)), 1)] ]:
                choices.append(x)
                weights.append(x.fitness if x.fitness > 0 else (- 1 / x.fitness) )

            if self.config.CROSS_SPECIE_REPRODUCTION:
                for x in [x for x in sorted(self.organisms, key=lambda f:f.fitness, reverse=True)[: max(math.floor(0.2 * len(s.organisms) ), 1)] ]:
                    if x.fitness > 0:
                        choices.append(x)
                        weights.append( x.fitness * self.config.CROSS_SPECIE_WEIGHT_MODIFIER)

            while specie_counter < s.offspring:
                g1, g2 = random.choices(choices, weights=weights, k=2)
                new_population.append(crossover(g1,g2))
                specie_counter += 1

        while len(new_population) <self.config.size:
            counter += 1
            new_population.append(copy.deepcopy(self.default_genome))
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

        if self.config.DYNAMIC_THRESHOLD:
            if len(self.species) < self.config.specie_target:
                Specie.threshold -= self.config.threshold_step_size
            if len(self.species) > self.config.specie_target:
                Specie.threshold += self.config.threshold_step_size
        assert sum(len(s.organisms) for s in self.species) == len(self.organisms)

class Specie:
    def __init__(self, id, representative):
        self.id = id
        self.organisms = [representative]
        self.representative =  representative
        self.gens_since_improvement = 0
        self.average = 0
        self.total_adjusted_fitness = 0

    def add_organism(self, organism):
        self.organisms.append(organism)

    def calculate_adjusted_fitness(self):
        N = len(self.organisms)
        for o in self.organisms:
            o.adjusted_fitness = o.fitness / N

    def clear_members(self):
        self.organisms = []
    
    def calculate_stats(self):
        self.calculate_adjusted_fitness()
        self.total_adjusted_fitness = sum([o.adjusted_fitness for o in self.organisms])
    def __repr__(self):
        return f"Specie id:{self.id}, len:{len(self.organisms)}, avg f{round(self.average, 3)}, gens since imp {self.gens_since_improvement}, avg nodes {sum([len(o.nodes) for o in self.organisms]) / len(self.organisms)}, avg conns: {sum([len(o.connections) for o in self.organisms]) / len(self.organisms)}"
