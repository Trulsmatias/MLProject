from src.trym_tests.NEAT import Connection
from src.trym_tests.NEAT.run import increment_and_get_innovation_number
import random
import numpy as np

class Genome:
    def __init__(self, nodes, input_nodes, output_nodes):
        self.connection_genes = []  # List of Connections between two Nodes
        self.fitness = 0
        self.adjusted_fitness = 0
        self.nodes = nodes  # List of all nodes, just numbers
        self.input_nodes = input_nodes  # Number of input nodes
        self.output_nodes = output_nodes  # Number of output nodes

    def add_connection(self):
        """
        Creates a new connection between two existing Nodes.
        The new connection can only go "forwards" in the network.
        input nodes can get connections to hidden nodes and output nodes, and
        hidden nodes can only get connections to output nodes.
        """
        if len(self.connection_genes) != 0:
            connection_exists = True
            while connection_exists:

                if len(self.nodes) == self.input_nodes + self.output_nodes:
                    input_node = random.choice(self.nodes[0:self.input_nodes])
                    input_type = 0
                else:
                    if random.uniform(0, 1) <= 0.7:
                        input_node = random.choice(self.nodes[0:self.input_nodes])
                        input_type = 0
                    else:
                        input_node = random.choice(self.nodes[self.input_nodes + self.output_nodes:])
                        input_type = 1

                if input_type == 0 and len(self.nodes) == self.input_nodes + self.output_nodes:
                    output_node = random.choice(self.nodes[self.input_nodes:])
                    output_type = 2

                elif input_type == 0:
                    if random.uniform(0, 1) <= 0.7:
                        output_node = random.choice(self.nodes[self.input_nodes:self.input_nodes + self.output_nodes])
                        output_type = 2
                    else:
                        output_node = random.choice(self.nodes[self.input_nodes + self.output_nodes:])
                        output_type = 1
                else:
                    output_node = random.choice(self.nodes[self.input_nodes:self.input_nodes + self.output_nodes])
                    output_type = 2

                connection_exists = False
                for con in self.connection_genes:
                    if input_node == con.in_node and output_node == con.out_node:
                        connection_exists = True
                        break

            if input_type == 0 and output_type == 1:
                type = 0
            elif input_type == 1 and output_type == 2:
                type = 1
            else:
                type = 2

            if input_node >= output_node and (type == 0 or type == 2):
                print('HER GIKK DET FAENMEG GALT!')
                print(input_node, output_type, type)

        else:
            input_node = random.choice(self.nodes[0:self.input_nodes])
            output_node = random.choice(self.nodes[self.input_nodes:])
            type = 2

        new_connection = Connection.Connection(input_node, output_node, type, random.uniform(-1, 1), True,
                                                increment_and_get_innovation_number(input_node, output_node))

        self.connection_genes.append(new_connection)

    def add_node(self):
        """
        Add a new hidden Node inside an existing connection.
        The new connection between old connection input and new node gets weight 1
        The new connection between new node and old connection output gets the weight of the old connection
        """
        new_node = len(self.nodes) + 1
        self.nodes = np.append(self.nodes, new_node)

        type_2_genes = [gene for gene in self.connection_genes if (gene.type == 2 and gene.enabled)]

        if (len(type_2_genes) == 0):
            print('LEN 0 FOR TYPE_2_GENES')
            for gene in self.connection_genes:
                print(gene.in_node, gene.out_node, gene.type, gene.enabled)

        else:
            old_connection = random.choice(type_2_genes)
            old_connection.set_disabled('add_node')

            new_connection1 = Connection.Connection(old_connection.in_node, new_node, 0, 1, True,
                                                    increment_and_get_innovation_number(old_connection.in_node,
                                                                                        new_node))

            new_connection2 = Connection.Connection(new_node, old_connection.out_node, 1,
                                                    old_connection.weight, True,
                                                    increment_and_get_innovation_number(new_node,
                                                                                        old_connection.out_node))

            self.connection_genes.append(new_connection1)
            self.connection_genes.append(new_connection2)

    def calculate_action(self, state, input_size, output_size):
        state = np.array(state).flatten()
        hidden_nodes = len(self.nodes) - (self.input_nodes + self.output_nodes)
        hidden_nodes = np.zeros(hidden_nodes)
        output_nodes = np.zeros(output_size)

        type_0_genes = [gene for gene in self.connection_genes if gene.type == 0 and gene.enabled]
        type_1_genes = [gene for gene in self.connection_genes if gene.type == 1 and gene.enabled]
        type_2_genes = [gene for gene in self.connection_genes if gene.type == 2 and gene.enabled]

        for con_gene in type_0_genes:
            in_value = state[int(con_gene.in_node - 1)]
            in_value *= con_gene.weight
            hidden_nodes[int(con_gene.out_node - (input_size + output_size + 1))] += in_value

        for con_gene in type_2_genes:
            in_value = state[int(con_gene.in_node - 1)]
            in_value *= con_gene.weight
            output_nodes[int(con_gene.out_node - (input_size + 1))] += in_value

        for con_gene in type_1_genes:
            in_value = hidden_nodes[int(con_gene.in_node - (input_size + output_size + 1))]
            in_value *= con_gene.weight
            output_nodes[int(con_gene.out_node - (input_size + 1))] += in_value

        # Compute softmax values for each sets of scores in output.
        output_nodes = np.exp(output_nodes - np.max(output_nodes))
        return output_nodes / output_nodes.sum(axis=0)
