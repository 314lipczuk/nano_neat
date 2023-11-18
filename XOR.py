import copy
import math
import random
import networkx as nx
from pyvis.network import Network as VisualNetwork

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
    def get_node_id():
        tmp = Node.next_node_id
        Node.next_node_id+=1
        return tmp

class Connection:
    next_innov_id=0

    def __init__(self, innov_id, in_node, out_node, weight=1.0, enabled=True, is_recurrent=False):
        self.innov_id=innov_id
        self.in_node=in_node
        self.out_node=out_node
        self.weight=weight
        self.enabled=enabled
        self.is_recurrent=is_recurrent
    
    @staticmethod
    def get_innov_id():
        tmp = Connection.next_innov_id
        Connection.next_innov_id+=1
        return tmp

class Genome:
    C1 = 1
    C2 = 1
    C3 = 0.4

    def activation_function(self, x):
        return 1 / (1+math.exp(-x))

    def __init__(self, inputN, outputN):
        self.nodes = []
        self.connections = []
        for _ in range(inputN):
            self.nodes.append(Node(node_id=Node.get_node_id(), node_type='input', layer=0))
        for _ in range(outputN):
            self.nodes.append(Node(node_id=Node.get_node_id(), node_type='output', layer=1))
        
        for i in [i for i in self.nodes if i.node_type == 'input']:
            for o in [o for o in self.nodes if o.node_type == 'output']:
                self.connections.append(Connection(innov_id=Connection.get_innov_id(), in_node=i.node_id, out_node=o.node_id, weight=random.random()))
        self.refresh_layers()
    
    def refresh_layers(self):
        inputs = [n.node_id for n in self.nodes if n.node_type == 'input']
        def find_max_len_to_input(node_id):
            conns = [1 if c.in_node in inputs else 1+find_max_len_to_input(c.in_node) for c in self.connections if c.enabled and not c.is_recurrent and c.out_node == node_id]
            return max(conns, default=0)
            
        for n in [n for n in self.nodes if n.node_type != 'input']:
            n.layer = find_max_len_to_input(n.node_id)

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
        for n, i in zip(list(sorted([n for n in self.nodes if n.node_type == 'input' ], key=lambda x: x.node_id)), inputs):
            n.sum_output = i
            n.sum_input = i

    def run_network(self):
        layers = sorted(list(set([n.node_layer for n in self.nodes])))
        for l in layers[1:]:
            nodes = [n for n in self.nodes if n.node_layer == l]
            for n in nodes:
                conns = [c for c in self.connections if c.out_node == n.node_id and c.enabled]
                n.sum_input = sum(c.weight * [nd for nd in self.nodes if c.in_node == nd.node_id][0].sum_output for c in conns)
                n.sum_output = self.activation_function(n.sum_input)
    def get_output(self):   
        return [n.sum_output for n in g.nodes if n.node_type == 'output']

    def compatibility_distance(self, other):
        n1, n2 = max([n.innov_id for n in self.connections if n.enabled]), max([n.innov_id for n in other.connections if n.enabled])
        N = max(n1,n2)
        excess_count = len([n for n in self.connections if n.innov_id > N and n.enabled]) + len([n for n in other.connections if n.innov_id > N and n.enabled])
        disjoint_count = len([ n for n in self.connections if n.innov_id <= N and n.enabled and n.innov_id not in [c.innov_id for c in other.connections if c.enabled]])\
            + len([ n for n in other.connections if n.innov_id <= N and n.enabled and n.innov_id not in [c.innov_id for c in self.connections if c.enabled]])

        common = [n.innov_id for n in self.connections if n.innov_id <= N and n.enabled and n.innov_id in [c.innov_id for c in other.connections if c.enabled]]
        a = [n.weight for n in sorted(self.connections, key=lambda x: x.innov_id) if n.innov_id in common]
        b = [n.weight for n in sorted(other.connections, key=lambda x: x.innov_id) if n.innov_id in common]

        weight_diff = sum([ abs(a[i]-b[i]) for i in range(len(a))]) / len(a)
        return Genome.C1 * excess_count + Genome.C2 * disjoint_count + Genome.C3 * weight_diff


class Population:
    size = 50
    def __init__(self, size):
        self.organisms = []
        self.species = []

class Specie:
    threshold = 3
    def __init__(self, id):
        self.id = id
        self.organisms = []
        self.best_organism = None
        self.best_fitness = 0
        self.average_fitness = 0
        self.staleness = 0
    

g = Genome(2,1)
print(g.nodes)
g.show()
g.load_inputs([0,1])
g.run_network()
print([n.sum_output for n in g.nodes if n.node_type == 'output'])
g1 = copy.deepcopy(g)
print('compd dist', g.compatibility_distance(g1))