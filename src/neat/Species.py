import random

class Species:
    def __init__(self, genome):
        self.species_representative = genome
        self.genomes = [genome]
        self.stagnant_generations = 0
        self.last_gen_best_fitness = 0

    def replace_representative(self):
        self.species_representative = random.choice(self.genomes)

    def update_species_parameters(self):
        highest_fitness = 0
        for genome in self.genomes:
            if genome.fitness > highest_fitness:
                highest_fitness = genome.fitness

        if highest_fitness <= self.last_gen_best_fitness:
            self.stagnant_generations += 1
        else:
            self.stagnant_generations = 0

        self.last_gen_best_fitness = highest_fitness
