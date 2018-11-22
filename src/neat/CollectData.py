import math
import csv
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os
import errno
import yaml
from util import get_path_of


class DataCollection:
    def __init__(self, population_size, generations, config,
                 path='saved_data/',
                 new_file_name=''):

        #self.simulation_params = simulation_params
        self.population_size = population_size
        self.generations = generations
        self.self_made_file_name = False
        self.new_file_name = new_file_name
        self.path = get_path_of(path)

        if self.new_file_name != '':
            self.self_made_file_name = True

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

        counter = 1
        # new_path = ""
        while True:

            if not self.self_made_file_name:
                self.new_file_name = 'graph.txt'
            new_path = self.path + "result" + str(counter) + "/"

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

                    file_obj.write(str(self.population_size) + ';' +
                                   str(self.generations) + '\n')
                    break

            if self.self_made_file_name:
                if self.new_file_name == new_file_name:
                    self.new_file_name = self.new_file_name.split('.')[0] + '_' + str(counter) + '.txt'
                else:
                    self.new_file_name = self.new_file_name.split('.')[0][:-1] + str(counter) + '.txt'

            counter += 1
        self.path = new_path

        with open(os.path.join(self.path, 'neat_params.yml'), 'w+') as file:
            yaml.dump(config, file, default_flow_style=False)

    def collect_data(self, genomes_list, gen_number, top_n_percent=10):

        gen_number = gen_number
        genomes_list = sorted(genomes_list, key=lambda genome: genome.fitness, reverse=True)  # Sort children by fitness

        best_genome = genomes_list[0]
        best_fitness = best_genome.fitness
        average_fitness = 0
        top_n_average_fitness = 0

        for genome in genomes_list:
            average_fitness += genome.fitness

        average_fitness = int(average_fitness / len(genomes_list))

        top_n_genomes = genomes_list[0:math.ceil((top_n_percent / 100) * len(genomes_list))]

        for genome in top_n_genomes:
            top_n_average_fitness += genome.fitness

        top_n_average_fitness = int(top_n_average_fitness / len(top_n_genomes))

        # write to graphfile
        with open(self.path + self.new_file_name, 'a') as file:
            file.write(str(gen_number) + ';' + str(best_fitness) + ';' + str(average_fitness) + ';' + str(
                top_n_average_fitness) + '\n')

        # Save best genome model
        with open(self.path + 'model_gen{}.obj'.format(gen_number), 'wb+') as file:
            pickle.dump(best_genome, file)


def read_csv(path):

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        buffer = [row for row in list(csv_reader)]

        individuals_per_gen, gens = buffer[0][0], buffer[0][1]
        a = buffer.pop(0)

        gen_numbers = [int(row[0]) for row in buffer]
        best_fitness = [int(row[1]) for row in buffer]
        average_fitness = [int(row[2]) for row in buffer]
        top_n_fitness = [int(row[3]) for row in buffer]

        return individuals_per_gen, gens, \
            gen_numbers, best_fitness, average_fitness, top_n_fitness


def make_graph(path):
    individuals_per_gen, gens, \
        gen_numbers, best_fitness, avereage_fitness, top_n_fitness = read_csv(path)

    df = pd.DataFrame({'x': gen_numbers, 'best': best_fitness,
                       'avg': avereage_fitness, 'top n': top_n_fitness})

    plt.plot('x', 'best', data=df, marker='', color='blue', linewidth=2)
    plt.plot('x', 'avg', data=df, marker='', color='red', linewidth=2)
    plt.plot('x', 'top n', data=df, marker='', color='green', linewidth=2)
    plt.figtext(0.125, 0.9, 'Specimen per gen = ' + str(individuals_per_gen), fontsize=12)
    plt.figtext(0.425, 0.9, 'Selected per gen = ' + str(gens), fontsize=12)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    make_graph('../saved_data/result1/graph.txt')