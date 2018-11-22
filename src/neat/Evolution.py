import random
import numpy as np
import math
from neat import Genome, Speciation, Connection
from itertools import combinations as comb


CREATE_NEW_CONNECTION_PROBABILITY = 1.0
CREATE_NEW_NODE_PROBABILITY = 0.2
MUTATE_NEW_CHILD_PROBABILITY = 0.7
MUTATE_GENE_PROBABILITY = 0.6
CHANGE_WEIGHT_SLIGHTLY_PROBABILITY = 0.9
KEEP_PERCENT_FROM_SPECIES = 0.7
KEEP_CONNECTION_DISABLED_PROBABILITY = 0.75


def set_globals_from_config(config):
    global CREATE_NEW_CONNECTION_PROBABILITY, CREATE_NEW_NODE_PROBABILITY, MUTATE_NEW_CHILD_PROBABILITY, \
        MUTATE_GENE_PROBABILITY, CHANGE_WEIGHT_SLIGHTLY_PROBABILITY, KEEP_PERCENT_FROM_SPECIES, \
        KEEP_CONNECTION_DISABLED_PROBABILITY
    CREATE_NEW_CONNECTION_PROBABILITY = config['create_new_connection_probability']
    CREATE_NEW_NODE_PROBABILITY = config['create_new_node_probability']
    MUTATE_NEW_CHILD_PROBABILITY = config['mutate_new_child_probability']
    MUTATE_GENE_PROBABILITY = config['mutate_gene_probability']
    CHANGE_WEIGHT_SLIGHTLY_PROBABILITY = config['change_weight_slightly_probability']
    KEEP_PERCENT_FROM_SPECIES = config['keep_percent_from_species']
    KEEP_CONNECTION_DISABLED_PROBABILITY = config['keep_connection_disabled_probability']


def make_new_generation(population_size, species_table, new_generation_num):
    Speciation.calculate_adjusted_fitness(species_table)
    reproduction_table = Speciation.reproduction_number_per_species(species_table, population_size)

    species_counter = 0

    new_children = []

    for species in species_table:
        sorted_genomes = sorted(species.genomes, key=lambda genome: genome.adjusted_fitness, reverse=True)
        new_children_from_species = make_children(reproduction_table[species_counter], sorted_genomes)
        new_children = np.concatenate((new_children, new_children_from_species), axis=None)

        species_counter += 1

    for i in range(len(new_children)):
        new_children[i].id = i + 1
        new_children[i].gen_num = new_generation_num

    return new_children


def make_children(number_of_children, species):

    best_from_species = species[0:math.ceil(len(species) * KEEP_PERCENT_FROM_SPECIES)]
    families = list(comb(best_from_species, 2))  # Every combination of families
    new_children = []

    """ Save the genome with the best fitness from a species if 
        number of genomes in species is greater than 5. """
    if len(species) > 5:
        species_champion = species[0]
        species_champion.gen_num += 1
        number_of_children -= 1
        new_children.append(species_champion)

    families = families[0:number_of_children]

    if len(best_from_species) < 2:
        for i in range(number_of_children):
            for genome in best_from_species:
                new_children.append(make_child(genome, genome))

    elif len(families) < number_of_children:
        full_times = math.floor(number_of_children / len(families))
        extra_children = number_of_children % len(families)

        for i in range(full_times):
            for family in families:
                new_children.append(make_child(family[0], family[1]))

        for i in range(extra_children):
            new_children.append(make_child(families[i][0], families[i][1]))

    else:
        for family in families:
            new_children.append(make_child(family[0], family[1]))


    return new_children


def make_child(genome1, genome2):

    if genome1.fitness > genome2.fitness:
        new_genes = select_genes(genome1, genome2)
    elif genome1.fitness < genome2.fitness:
        new_genes = select_genes(genome2, genome1)
    else:
        new_genes = select_genes(genome1, genome2)

    nodes = [i + 1 for i in range(genome1.input_nodes + genome1.output_nodes)]
    hidden_nodes = []

    for gene in new_genes:
        if gene.type == 0:
            if gene.out_node not in hidden_nodes:
                hidden_nodes.append(gene.out_node)
        elif gene.type == 1:
           if gene.in_node not in hidden_nodes:
               hidden_nodes.append(gene.in_node)

    node_array = np.concatenate((nodes, hidden_nodes), axis=None)

    new_genome = Genome.Genome(0, 0, node_array, genome1.input_nodes, genome1.output_nodes)
    new_genome.connection_genes = new_genes

    """ Add a new connection between to nodes"""
    if random.uniform(0, 1) <= CREATE_NEW_CONNECTION_PROBABILITY:
        new_genome.add_connection()

    """ Add a new node inside an existing connection between input node and output node """
    if random.uniform(0, 1) <= CREATE_NEW_NODE_PROBABILITY:
        new_genome.add_node()

    """ Randomly mutate som genes """
    if random.uniform(0, 1) <= MUTATE_NEW_CHILD_PROBABILITY:
        for genome in new_genome.connection_genes:
            if random.uniform(0, 1) <= MUTATE_GENE_PROBABILITY:
                if random.uniform(0, 1) <= CHANGE_WEIGHT_SLIGHTLY_PROBABILITY:
                    genome.weight = genome.weight + random.uniform(-1, 1) * 0.1
                else:
                    genome.weight = random.uniform(-1, 1)

    return new_genome


def select_genes(fittest_parent, weakest_parent):
    new_genes = []

    fp_genes = [con.innovation_number for con in fittest_parent.connection_genes]
    wp_genes = [con.innovation_number for con in weakest_parent.connection_genes]

    equal_genes = frozenset(fp_genes).intersection(wp_genes)

    for i in equal_genes:
        if random.randint(0, 1) == 0:
            for con in fittest_parent.connection_genes:
                if i == con.innovation_number:
                    new_con = Connection.Connection(con.in_node, con.out_node, con.type,
                                                    con.weight, con.enabled, con.innovation_number)

                    if not new_con.enabled:
                        if random.uniform(0, 1) >= KEEP_CONNECTION_DISABLED_PROBABILITY:
                            new_con.enabled = True

                    new_genes.append(new_con)
        else:
            for con in weakest_parent.connection_genes:
                if i == con.innovation_number:
                    new_con = Connection.Connection(con.in_node, con.out_node, con.type,
                                                    con.weight, con.enabled, con.innovation_number)

                    if not new_con.enabled:
                        if random.uniform(0, 1) >= KEEP_CONNECTION_DISABLED_PROBABILITY:
                            new_con.enabled = True

                    new_genes.append(new_con)

    unique_genes = np.setdiff1d(fp_genes, wp_genes)

    for i in unique_genes:
        for con in fittest_parent.connection_genes:
            if i == con.innovation_number:
                new_con = Connection.Connection(con.in_node, con.out_node, con.type,
                                                con.weight, con.enabled, con.innovation_number)

                if not new_con.enabled:
                    if random.uniform(0, 1) >= KEEP_CONNECTION_DISABLED_PROBABILITY:
                        new_con.enabled = True

                new_genes.append(new_con)

    return new_genes


if __name__ == '__main__':
    a = [2]
    families = list(comb(a, 2))  # Every combination of families

    print(a)
    print(families)
