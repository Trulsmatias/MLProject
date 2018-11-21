from src.trym_tests.NEAT import Genome, Speciation, SimulateMario, Evolution, CollectData

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
    global mutations_in_gen
    population_size = 300
    generations = 50
    species_list = []
    genomes_list = []
    input_nodes = 130
    output_nodes = 7
    initial_nodes = [i + 1 for i in range(input_nodes + output_nodes)]

    new_genome = Genome.Genome(initial_nodes, input_nodes, output_nodes)
    new_genome.add_connection()
    new_genome.add_connection()
    new_genome.add_connection()
    new_genome.add_connection()

    species_list.append([new_genome])
    genomes_list.append(new_genome)

    data = CollectData.DataCollection(population_size, generations)

    for i in range(population_size - 1):

        new_genome = Genome.Genome(initial_nodes, input_nodes, output_nodes)
        new_genome.add_connection()
        new_genome.add_connection()
        new_genome.add_connection()
        new_genome.add_connection()
        Speciation.add_to_species(species_list, new_genome)
        genomes_list.append(new_genome)



    print('New genomes added to species list:', input_nodes + output_nodes)
    print('Number of new species:', len(species_list))

    for gen in range(generations):
        mutations_in_gen = []
        counter = 0
        for genome in genomes_list:
            SimulateMario.simulate_run(genome, 5000, False)
            counter += 1
            print('Simulating genome:', counter)

        data.collect_data(genomes_list, gen + 1)

        print('\nFitness: ')
        for genome in genomes_list:
            print(genome.fitness)

        print('\nMaking new children.')
        print('\nNumber of species:', len(species_list))

        Speciation.calculate_adjusted_fitness(species_list)
        genomes_list = Evolution.make_new_generation(population_size, species_list)
        species_list = [[genomes_list[0]]]

        for i in range(1, len(genomes_list)):
            Speciation.add_to_species(species_list, genomes_list[i])

        print('Generation', gen + 1, 'done.\n')





if __name__ == '__main__':
    run_simulation()
