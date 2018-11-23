from naive import generations, agent
from naive.parallel.task import PickleSimulationTask
import numpy as np
import logging

_log = logging.getLogger('MLProject.parallel.serialize')


def pickle_individual_to_task(individual):
    weights = individual.agent.get_weights()
    return PickleSimulationTask(individual.id, weights,
                                individual.agent.state_space_shape, individual.agent.action_space_size)


def pickle_task_to_individual(task):
    individual = generations.Individual(agent.NNAgent(task.state_space_shape, task.action_space_size))
    individual.id = task.id
    _log.debug('Created pseudo-individual {}'.format(individual))

    _log.debug('Setting weights for NNAgent, shape {}'.format([np.shape(w) for w in task.weights]))
    individual.agent.set_weights(task.weights)
    return individual
