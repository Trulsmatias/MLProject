import logging
import signal
import sys
from naive.logger import setup_logging
import multiprocessing as mp
import os
import queue
from naive import play
from naive.parallel.serialize import pickle_task_to_individual
from naive.parallel.task import SimulationResult


def worker_proc(worker_num, task_queue: mp.JoinableQueue, result_queue: mp.Queue,
                shutdown_event: mp.Event, simulator_params):

    # Create synchronous Simulator
    # Loop:
    #   If queue not empty, wait max 1s:
    #     Pull item from queue, simulate, set fitness
    #   If shutdown event is set:
    #     Shutdown and return

    setup_logging(simulator_params['log_level'])
    log = logging.getLogger('MLProject.parallel.Worker_{}'.format(worker_num))

    def shutdown(sig=0, sframe=None):
        log.info('Worker shutting down')
        sys.exit(0)

    def shutdown_ignore(sig=0, sframe=None):
        pass

    signal.signal(signal.SIGTERM, shutdown_ignore)
    signal.signal(signal.SIGINT, shutdown_ignore)

    log.info('Worker started. PID: {}. Parent PID: {}'.format(os.getpid(), os.getppid()))

    simulator = play.Simulator(simulator_params['movements'], simulator_params['max_steps'])

    while True:
        try:
            log.debug('Checking for tasks')
            task = task_queue.get(timeout=1)

            log.debug('Got task for individual {}'.format(task.id))
            individual = pickle_task_to_individual(task)

            log.info('Simulating individual {}'.format(individual.id))
            simulator._simulate_individual(individual, render=simulator_params['render'])

            result_queue.put(SimulationResult(individual.id, individual.fitness))
            task_queue.task_done()

        except queue.Empty:
            log.debug('Task queue is empty')

        if shutdown_event.is_set():
            shutdown()
            return
