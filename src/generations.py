class Individual:
    def __init__(self, id, agent):
        self.id = id
        self.fitness = 0
        self.agent = agent

    def __repr__(self):
        return 'Individual(id={}, fitness={}, agent={})'.format(self.id, self.fitness, self.agent)


class Generation:
    def __init__(self, num, individuals):
        """
        Creates a new Generation.
        :param num: the generation number.
        :param individuals: a list of this generations' individuals
        """
        self.num = num
        self.individuals = individuals

    def __repr__(self):
        individuals_str = ''
        if len(self.individuals) > 0:
            individuals_str = '\n    ' + '\n    '.join([str(i) for i in self.individuals]) + '\n'
        return 'Generaiton(num={}, children={})'.format(self.num, individuals_str)


class EvolutionParameters:
    def __init__(self, selection_func, num_parents_per_child, breeding_func, mutation_rate, num_select):
        """
        Defines parameters which controls evolution of generations.
        :param selection_func: the selection function to use
        :param num_parents_per_child:
        :param breeding_func: a function which creates a single child given a list of parents
        :param mutation_rate:
        :param num_select:
        """
        self.selection_func = selection_func
        self.num_parents_per_child = num_parents_per_child
        self.breeding_func = breeding_func
        self.mutation_rate = mutation_rate
        self.num_select = num_select
