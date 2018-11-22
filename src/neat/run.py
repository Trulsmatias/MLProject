import multiprocessing
import random
from neat import Genome, Speciation, Evolution, CollectData, Species
from neat.config import load_config, setup_logging
from neat.parallel.simulate import ParallelSimulator

innovation_number = 0
mutations_in_gen = []


def increment_and_get_innovation_number(input_node, output_node):
    global innovation_number

    for mutation in mutations_in_gen:
        if mutation.input_node == input_node and mutation.output_node == output_node:
            return mutation.innovation_number

    innovation_number += 1
    mutations_in_gen.append(Mutation(input_node, output_node, innovation_number))
    return innovation_number


class Mutation:
    def __init__(self, input_node, output_node, innovation_number):
        self.input_node = input_node
        self.output_node = output_node
        self.innovation_number = innovation_number


def run_simulation():
    multiprocessing.set_start_method('spawn')  # Spawn fresh worker processes, don't use fork

    config = load_config()
    setup_logging(config['log_level'])
    Evolution.set_globals_from_config(config)
    Speciation._config = config

    print('Using these parameters:')
    for k, v in config.items():
        print('- {}: {}'.format(k, v))

    global mutations_in_gen
    population_size = config['population_size']  # default 500
    generations = config['generations']  # default 100
    species_list = []
    genomes_list = []
    input_nodes = config['input_nodes']  # default 130
    output_nodes = config['output_nodes']  # default 7
    initial_nodes = [i + 1 for i in range(input_nodes + output_nodes)]

    new_genome = Genome.Genome(1, 1, initial_nodes, input_nodes, output_nodes)

    number_of_new_connections = random.randint(2, 8)
    for i in range(number_of_new_connections):
        new_genome.add_connection()


    new_species = Species.Species(new_genome)
    species_list.append(new_species)

    genomes_list.append(new_genome)

    data = CollectData.DataCollection(population_size, generations, config)

    for i in range(population_size - 1):

        new_genome = Genome.Genome(i + 2, 1, initial_nodes, input_nodes, output_nodes)
        number_of_new_connections = random.randint(1, 4)

        for _ in range(number_of_new_connections):
            new_genome.add_connection()

        Speciation.add_to_species(species_list, new_genome)
        genomes_list.append(new_genome)


    print('New genomes added to species list:', input_nodes + output_nodes)
    print('Number of new species:', len(species_list))

    simulator = ParallelSimulator(num_workers=config['num_workers'],
                                  max_steps=config['max_simulation_steps'],
                                  render=config['render'])
    for gen_num in range(1, generations + 1):
        mutations_in_gen = []
        """
        counter = 0
        for genome in genomes_list:
            SimulateMario.simulate_run(genome, 5000, False)
            counter += 1
            print('Simulating genome:', counter)
        """
        simulator.simulate_genomes(genomes_list)

        data.collect_data(genomes_list, gen_num, len(species_list))

        print('\nFitness:', [genome.fitness for genome in genomes_list])
        # for genome in genomes_list:
        #     print(genome.fitness)

        print('\nMaking new children.')
        print('\nNumber of species:', len(species_list))

        Speciation.calculate_adjusted_fitness(species_list)
        genomes_list = Evolution.make_new_generation(population_size, species_list, gen_num + 1)

        """ Empty all genomes from species """
        for species in species_list:
            # species.update_species_parameters()
            species.genomes = []

        """ Add new genomes to the existing species, or create new ones """
        for i in range(0, len(genomes_list)):
            Speciation.add_to_species(species_list, genomes_list[i])

        print('Generation', gen_num, 'done.\n')
        """ Remove all species without genomes """
        species_without_genomes_exist = True
        while species_without_genomes_exist:
            species_without_genomes_exist = False

            for i in range(0, len(species_list)):
                if len(species_list[i].genomes) == 0:
                    del species_list[i]
                    species_without_genomes_exist = True
                    break

        """ Replace old species representative with new """
        for species in species_list:
            species.replace_representative()


if __name__ == '__main__':
    run_simulation()
