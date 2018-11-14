import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os
import errno


class DataCollection:
    def __init__(self, simulation_params,
                 path='../saved_data/',
                 new_file_name=''):

        #self.simulation_params = simulation_params
        self.population = simulation_params.num_individuals_per_gen
        self.selected_per_gen = simulation_params.num_select
        self.mutation_rate_individual = simulation_params.mutation_rate_individual
        self.mutation_rate_genes = simulation_params.mutation_rate_genes
        self.self_made_file_name = False
        self.new_file_name = new_file_name
        self.path = path

        if self.new_file_name != '':
            self.self_made_file_name = True

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

        counter = 1
        # new_path = ""
        while True:

            if not self.self_made_file_name:
                self.new_file_name = 'graph.txt'
            new_path = path + "result" + str(counter) + "/"

            if not os.path.exists(new_path):
                os.makedirs(new_path)
            try:
                file_handle = os.open(new_path + self.new_file_name, flags)
            except OSError as e:
                if e.errno == errno.EEXIST:  # Failed as the file already exists.
                    pass
                else:  # Something unexpected went wrong so reraise the exception.
                    raise
            else:  # No exception, so the file must have been created successfully.
                with os.fdopen(file_handle, 'w') as file_obj:

                    file_obj.write(str(self.population) + ';' +
                                   str(self.selected_per_gen) + ';' +
                                   str(self.mutation_rate_individual) + ';' +
                                   str(self.mutation_rate_genes) + '\n')
                    break

            if self.self_made_file_name:
                if self.new_file_name == new_file_name:
                    self.new_file_name = self.new_file_name.split('.')[0] + '_' + str(counter) + '.txt'
                else:
                    self.new_file_name = self.new_file_name.split('.')[0][:-1] + str(counter) + '.txt'

            counter += 1
        self.path = new_path

    def collect_data(self, generation, top_n_percent=10):

        gen_number = generation.num
        gen_children = generation.individuals
        gen_children = sorted(gen_children, key=lambda child: child.fitness, reverse=True)  # Sort children by fitness

        best_fitness = gen_children[0].fitness
        average_fitness = 0
        top_n_average_fitness = 0

        for child in gen_children:
            average_fitness += child.fitness

        average_fitness = int(average_fitness / len(gen_children))

        top_n_children = gen_children[0:math.ceil((top_n_percent / 100) * len(gen_children))]

        for child in top_n_children:
            top_n_average_fitness += child.fitness

        top_n_average_fitness = int(top_n_average_fitness / len(top_n_children))

        # write to graphfile
        file = open(self.path + self.new_file_name, 'a')
        file.write(str(gen_number) + ';' + str(best_fitness) + ';' + str(average_fitness) + ';' + str(
            top_n_average_fitness) + '\n')

    def save_module(self, individual):
        # write to model file
        individual.agent.save_model(self.path + "model.h5")


def read_csv(path):

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        buffer = [row for row in list(csv_reader)]

        individuals_per_gen, selected_per_gen, mutation_rate = buffer[0][0], buffer[0][1], buffer[0][2]
        a = buffer.pop(0)

        gen_numbers = [int(row[0]) for row in buffer]
        best_fitness = [int(row[1]) for row in buffer]
        average_fitness = [int(row[2]) for row in buffer]
        top_n_fitness = [int(row[3]) for row in buffer]

        return individuals_per_gen, selected_per_gen, mutation_rate, \
            gen_numbers, best_fitness, average_fitness, top_n_fitness


def make_graph(path):
    individuals_per_gen, selected_per_gen, mutation_rate, \
        gen_numbers, best_fitness, avereage_fitness, top_n_fitness = read_csv(path)

    df = pd.DataFrame({'x': gen_numbers, 'best': best_fitness,
                       'avg': avereage_fitness, 'top n': top_n_fitness})

    plt.plot('x', 'best', data=df, marker='', color='blue', linewidth=2)
    plt.plot('x', 'avg', data=df, marker='', color='red', linewidth=2)
    plt.plot('x', 'top n', data=df, marker='', color='green', linewidth=2)
    plt.figtext(0.125, 0.9, 'Specimen per gen = ' + str(individuals_per_gen), fontsize=12)
    plt.figtext(0.425, 0.9, 'Selected per gen = ' + str(selected_per_gen), fontsize=12)
    plt.figtext(0.725, 0.9, 'Mutation rate = ' + str(mutation_rate), fontsize=12)
    plt.legend()
    plt.show()