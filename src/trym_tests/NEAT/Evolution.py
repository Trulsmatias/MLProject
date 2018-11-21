import random
import numpy as np
import math
from src.trym_tests.NEAT import Genome, Speciation, Connection
from itertools import combinations as comb


def make_new_generation(population_size, species_table):
    Speciation.calculate_adjusted_fitness(species_table)
    reproduction_table = Speciation.reproduction_number_per_species(species_table, population_size)

    species_counter = 0

    new_children = []

    for species in species_table:
        species = sorted(species, key=lambda genome: genome.adjusted_fitness, reverse=True)
        new_children_from_species = make_children(reproduction_table[species_counter], species)
        new_children = np.concatenate((new_children, new_children_from_species), axis=None)

        species_counter += 1

    return new_children


def make_children(number_of_children, species):
    best_from_species = species[0:math.ceil(len(species) / 2)]
    families = list(comb(best_from_species, 2))  # Every combination of families
    families = families[0:number_of_children]

    new_children = []

    if len(families) < 1:
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

    nodes = [i + 1 for i in range(137)]
    hidden_nodes = []

    for gene in new_genes:
        if gene.type == 0:
            if gene.out_node not in hidden_nodes:
                hidden_nodes.append(gene.out_node)
        elif gene.type == 1:
           if gene.in_node not in hidden_nodes:
               hidden_nodes.append(gene.in_node)

    node_array = np.concatenate((nodes, hidden_nodes), axis=None)
    input_nodes = 130
    output_nodes = 7

    new_genome = Genome.Genome(node_array, input_nodes, output_nodes)
    new_genome.connection_genes = new_genes

    """ Add a new connection between to nodes"""
    if random.uniform(0, 1) <= 1.0:
        new_genome.add_connection()

    """ Add a new node inside an existing connection between input node and output node """
    if random.uniform(0, 1) <= 0.2:
        new_genome.add_node()

    """ Randomly mutate som genes """
    if random.uniform(0, 1) <= 0.7:
        for genome in new_genome.connection_genes:
            if random.uniform(0, 1) <= 0.8:
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
                    new_genes.append(new_con)
        else:
            for con in weakest_parent.connection_genes:
                if i == con.innovation_number:
                    new_con = Connection.Connection(con.in_node, con.out_node, con.type,
                                                    con.weight, con.enabled, con.innovation_number)
                    new_genes.append(new_con)

    unique_genes = np.setdiff1d(fp_genes, wp_genes)

    for i in unique_genes:
        for con in fittest_parent.connection_genes:
            if i == con.innovation_number:
                new_con = Connection.Connection(con.in_node, con.out_node, con.type,
                                                con.weight, con.enabled, con.innovation_number)
                new_genes.append(new_con)

    return new_genes
