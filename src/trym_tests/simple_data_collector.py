import math


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

    file = open('test_run_1', 'a')
    file.write(str(gen_number) + ';' + str(best_fitness) + ';' + str(average_fitness) + ';' + str(top_n_average_fitness) + '\n')
