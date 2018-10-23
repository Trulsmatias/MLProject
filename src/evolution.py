from generations import Generation, Individual
import numpy as np


# TODO: Input a list of Individuals instead of the Generation object? In accordance with reproduce() and mutate()
def roulette_wheel_selection(generation, num_select):
    """
    Performs roulette wheel selection on individuals in the generation.
    :param generation:
    :param num_select: number of individuals to select
    :return: a list of individuals that were selected
    """
    individuals_sorted = sorted(generation.individuals,
                                key=lambda individual: individual.fitness,
                                reverse=True)
    fitness_sum = sum([individual.fitness for individual in individuals_sorted])
    probabilities = [individual.fitness / fitness_sum for individual in individuals_sorted]

    return np.random.choice(individuals_sorted, size=num_select, replace=False, p=probabilities)


def make_child(parents):
    """
    Make a single child from a list of parents.
    :param parents: a list of parent which will make a child
    :return: the child
    """
    # TODO: Cross-breed parents, now it only selects a random parent.
    # This is where cross-breeding of weights and biases of the parents' NNs should happen.
    # There's prob many ways to do this, maybe make several different versions
    return np.random.choice(parents)


def reproduce(parents, num_parents_per_child, num_children_total):
    """
    Creates a number of children by reproduction.
    :param parents: the parents, aka. the fittest individuals after selection
    :param num_parents_per_child:
    :param num_children_total:
    :return: children of the next generation
    """
    children = []
    for i in range(0, len(parents), num_parents_per_child):
        family_parents = parents[i:i+num_parents_per_child]
        num_children_per_family = num_children_total // (len(parents) // num_parents_per_child)
        for j in range(num_children_per_family):  # TODO: fix this monstrosity
            children.append(make_child(family_parents))

    return children


def mutate(children):
    """
    Mutates individuals (children).
    :param children: a list of Individuals (the children) to mutate
    :return: a list of mutated children
    """
    # TODO: Mutate the children (add randomness to weights/biases in the NNs).
    return children


if __name__ == '__main__':
    np.random.seed(42)

    individuals = []
    for i in range(12):
        chromosomes = np.random.random()
        individual = Individual(i + 1, chromosomes)
        individuals.append(individual)

    gen = Generation(1, individuals)

    for i in range(10):
        for individual in gen.individuals:
            individual.fitness = np.random.random()

        best = roulette_wheel_selection(gen, 4)
        children = reproduce(best, num_parents_per_child=2, num_children_total=12)
        children = mutate(children)

        print(gen)
        gen = Generation(gen.num + 1, children)

        # print(gen.individuals)
        # print(roulette_wheel_selection(gen, 1))
