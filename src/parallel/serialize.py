import sys
from multiprocessing.sharedctypes import RawArray
from parallel.task import SharedMemSimulationTask, PickleSimulationTask
import numpy as np
import logging
import agent
import generations

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


def sharedmem_individual_to_task(individual):
    _log.debug('Serializing individual {}'.format(individual.id))
    shared_arrays = []
    shapes = []
    for matrix in individual.agent.get_weights():
        shared_array = RawArray('d', matrix.size)  # TODO er 'd' alltid riktig datatype?
        np_shared_array = np.frombuffer(shared_array).reshape(matrix.shape)
        np.copyto(np_shared_array, matrix)
        _log.debug('Shared array buffer {}, elems {}, sizeof {}'.format(shared_array, matrix.size,
                                                                        sys.getsizeof(shared_array)))

        shared_arrays.append(shared_array)
        shapes.append(matrix.shape)

    return SharedMemSimulationTask(individual.id, shared_arrays, shapes,
                                   individual.agent.state_space_shape, individual.agent.action_space_size)


def sharedmem_task_to_individual(task):
    _log.debug('Deserializing individual {}'.format(task.id))
    individual = generations.Individual(agent.NNAgent(task.state_space_shape, task.action_space_size))
    individual.id = task.id

    weights = []
    for matrix, shape in zip(task.weights, task.shapes):
        weights.append(np.frombuffer(matrix).reshape(shape))
        _log.debug('Matrix buffer: {}'.format(matrix))
    individual.agent.set_weights(weights)

    return individual


def mmap_individual_to_task(individual):
    pass


def mmap_task_to_individual(task):
    pass
