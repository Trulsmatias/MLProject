from agent import NNAgent


class Individual:
    def __init__(self, agent):
        self.id = 0
        self.fitness = 0
        self.agent: NNAgent = agent

    def __repr__(self):
        return 'Individual(id={}, fitness={}, agent_id={})'.format(self.id, self.fitness, hex(id(self.agent)))


class Generation:
    def __init__(self, num, individuals):
        """
        Creates a new Generation.
        Creates and sets a individual ID for all individuals in this generation.
        :param num: the generation number.
        :param individuals: a list of this generations' individuals
        """
        self.num = num

        individual_num = 1
        for individual in individuals:
            individual.id = '{}-{}'.format(num, individual_num)
            individual_num += 1

        self.individuals = individuals

    def __repr__(self):
        individuals_str = ''
        if len(self.individuals) > 0:
            individuals_str = '\n    ' + '\n    '.join([str(i) for i in self.individuals]) + '\n'
        return 'Generaiton(num={}, children={})'.format(self.num, individuals_str)

    def add_individual(self, individual):
        individual.id = '{}-{}'.format(self.num, len(self.individuals) + 1)
        self.individuals.append(individual)


class SimulationParameters:
    def __init__(self, # state_space_shape, action_space_shape, max_simulation_steps, num_generations,
                 #num_individuals_per_gen,
                 selection_func, num_parents_per_child, breeding_func,
                 mutation_rate, num_select):
        """
        Defines parameters which controls evolution of generations.
        :param selection_func: the selection function to use
        :param num_parents_per_child:
        :param breeding_func: a function which creates a single child given a list of parents
        :param mutation_rate:
        :param num_select:
        """

        """
        self.state_space_shape = state_space_shape
        self.action_space_shape = action_space_shape
        self.max_simulation_steps = max_simulation_steps
        self.num_generations = num_generations
        self.num_individuals_per_gen = num_individuals_per_gen
        """
        self.selection_func = selection_func
        self.num_parents_per_child = num_parents_per_child
        self.breeding_func = breeding_func
        self.mutation_rate = mutation_rate
        self.num_select = num_select
