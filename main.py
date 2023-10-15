import typing
import math
import random

"""
XOR using neat
"""

class Genome:
    class Node:
        types = ['Input', 'Hidden', 'Output', 'Sensor']
        def __init__(self, type:tuple[str],num ,  active=True):
            self.inType= type[0]
            self.outType= type[1]
            self.num = num

    class Connection:
        def __init__(self,input, out, weight,enabled, innov):
            self.enabled= enabled
            self.input = input
            self.output = out
            self.weight = weight
            self.innov = innov

        def split_in_two(self, newNode:int, innov:int):
            in_conn = Genome.Connection(input=self.input, out=newNode, weight=1, enabled=True, innov= innov+1)
            out_conn = Genome.Connection(input=newNode, out=self.output, weight=self.weight, enabled=True, innov= innov+2)
            self.enabled = False
            return [in_conn, out_conn]

    def __init__(self):
        self.node_genes : dict[int, Genome.Node] = {}
        self.connection_genes : list[Genome.Connection] = []

    def mutate_add_node(self):
        random_connection = self.connection_genes[random.randint(0, len(self.connection_genes)-1)]
        new_node_init = (
            self.node_genes[random_connection.input].outType,
            self.node_genes[random_connection.output].inType
            )
        node_innov = max(self.node_genes.keys()) + 1
        self.node_genes[node_innov] = Genome.Node(new_node_init, num=node_innov)
        conn_innov = len(self.connection_genes) - 1
        new_conns = random_connection.split_in_two(newNode=node_innov, innov=conn_innov )
        self.connection_genes += new_conns

    def mutate_add_connection(self):
        choices_in =  list(filter(None,[i if i.inType != 'Output' else None for i in  self.node_genes.values()]))
        choices_out = list(filter(None,[i if i.inType != 'Input' else None for i in  self.node_genes.values()]))
        choice_in = random.choice(choices_in)
        choice_out = random.choice(filter(lambda i : i != choice_in, choices_out))
        innov = len(self.connection_genes) -1
        new_conn = Genome.Connection(input=choice_in, out=choice_out, weight=random.random(), enabled=True, innov=innov)
        self.connection_genes.append(new_conn)

    def add_node(self, node):
        i = max(self.node_genes.keys() or [0]) + 1 
        self.node_genes[i] = node

    @staticmethod
    def init_with_shape(shape:tuple[int]):
        g = Genome()
        type_tuple = ("Sensor", "Input")
        input_size, output_size = shape
        for i in range(input_size):
            n = Genome.Node(type_tuple)
            g.add_node(n)
        
