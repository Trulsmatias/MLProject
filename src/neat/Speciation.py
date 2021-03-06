import numpy as np
import math
from neat import Species

_config = None


def reproduction_number_per_species(species_table, population_size):

    species_fitness_table = []
    total_fitness_sum = 0
    for species in species_table:
        ad_fitness_sum = 0
        if species.stagnant_generations < 15:
            for genome in species.genomes:
                ad_fitness_sum += genome.adjusted_fitness

        total_fitness_sum += ad_fitness_sum
        species_fitness_table.append(ad_fitness_sum)

    species_fitness_table = [math.floor((i/total_fitness_sum) * population_size) for i in species_fitness_table]

    diff = population_size - np.sum(species_fitness_table)

    if diff != 0:
        highest_repro_number = 0
        highest_repro_index = -1
        for i in range(len(species_fitness_table)):
            if species_fitness_table[i] > 0 and species_fitness_table[i] > highest_repro_number:
                highest_repro_number = species_fitness_table[0]
                highest_repro_index = i

        species_fitness_table[highest_repro_index] += diff

    return species_fitness_table

def calculate_adjusted_fitness(species_table):
    for species in species_table:
        for genome in species.genomes:
            genome.adjusted_fitness = genome.fitness / len(species.genomes)

def add_to_species(species_table, new_genome):
    found_compatable_species = False
    for species in species_table:
        if is_compatible(species.species_representative, new_genome):
            species.genomes.append(new_genome)
            found_compatable_species = True
            break

    if not found_compatable_species:
        new_species = Species.Species(new_genome)
        species_table.append(new_species)

def is_compatible(genome1, genome2):
    g1_genes = [con.innovation_number for con in genome1.connection_genes]
    g2_genes = [con.innovation_number for con in genome2.connection_genes]

    comp_thresh = _config['comp_thresh']  # 4.0  # If comp is heigher than this number, the genomes are not compatible

    c1 = _config['c1']  # 1.0
    c3 = _config['c3']  # 4.0

    disjoint_and_excess_genes = len(np.setdiff1d(g1_genes, g2_genes)) + len(np.setdiff1d(g2_genes, g1_genes))

    equal_genes = frozenset(g1_genes).intersection(g2_genes)
    diff = 0
    for i in equal_genes:
        gene1_weight = 0
        gene2_weight = 0

        for gene1 in genome1.connection_genes:
            if i == gene1.innovation_number:
                gene1_weight = gene1.weight
                break

        for gene2 in genome2.connection_genes:
            if i == gene2.innovation_number:
                gene2_weight = gene2.weight
                break

        diff += abs(gene1_weight - gene2_weight)

    if (len(equal_genes) != 0):
        w = diff / len(equal_genes)
    else:
        w = 5

    n = 1

    if len(g1_genes) > len(g2_genes) and len(g1_genes) >= 20:
        n = len(g1_genes)
    elif len(g1_genes) <= len(g2_genes) and len(g2_genes) >= 20:
        n = len(g2_genes)

    return (((c1 * disjoint_and_excess_genes) / n) + (w * c3)) < comp_thresh

if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 8, 10, 7]
    b = [1, 2, 3, 4, 5, 6, 7, 9, 10]



    c = frozenset(b).intersection(a)
    for i in c:
        print(i)