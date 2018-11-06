import math
import csv
import matplotlib.pyplot as plt
import pandas as pd


def collect_data(generation, top_n_percent=5):

    gen_number = generation.num
    gen_children = generation.individuals
    gen_children = sorted(gen_children, key=lambda child: child.fitness, reverse=True)  # Sort children by fitness

    best_fitness = gen_children.pop(0).fitness
    average_fitness = 0
    top_n_average_fitness = 0

    for child in gen_children:
        average_fitness += child.fitness

    average_fitness = int(average_fitness / len(gen_children))

    top_n_children = gen_children[0:math.ceil((top_n_percent/100) * len(gen_children))]

    for child in top_n_children:
        top_n_average_fitness += child.fitness

    top_n_average_fitness = int(top_n_average_fitness / len(top_n_children))

    file = open('test_run_1.txt', 'a')
    file.write(str(gen_number) + ';' + str(best_fitness) + ';' + str(average_fitness) + ';' + str(top_n_average_fitness) + '\n')


def read_csv(file='../test_run_1.txt'):

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        buffer = [row for row in list(csv_reader)]

        gen_numbers = [int(row[0]) for row in buffer]
        best_fitness = [int(row[1]) for row in buffer]
        average_fitness = [int(row[2]) for row in buffer]
        top_n_fitness = [int(row[3]) for row in buffer]

        return gen_numbers, best_fitness, average_fitness, top_n_fitness


def make_graph():
    gen_numbers, best_fitness, avereage_fitness, top_n_fitness = read_csv()
    print(len(gen_numbers), len(best_fitness), len(avereage_fitness), len(top_n_fitness))

    df = pd.DataFrame({'x': gen_numbers, 'best': best_fitness,
                       'avg': avereage_fitness, 'top n': top_n_fitness})


    plt.plot('x', 'best', data=df, marker='', color='blue', linewidth=2)
    plt.plot('x', 'avg', data=df, marker='', color='red', linewidth=2)
    plt.plot('x', 'top n', data=df, marker='', color='green', linewidth=2)
    plt.legend()
    plt.show()