import logging
import mmap
import signal
import sys
from logger import setup_logging
import multiprocessing as mp
import os
import queue
import play
from parallel.serialize import sharedmem_task_to_individual, pickle_task_to_individual
from parallel.task import SimulationResult


def worker_proc(worker_num, task_queue: mp.JoinableQueue, result_queue: mp.Queue,
                shutdown_event: mp.Event, simlulator_params):

    # Create synchronous Simulator
    # Loop:
    #   If queue not empty, wait max 1s:
    #     Pull item from queue, simulate, set fitness
    #   If shutdown event is set:
    #     Shutdown and return

    setup_logging()
    log = logging.getLogger('MLProject.parallel.Worker_{}'.format(worker_num))

    # memmap_fd = open('mmap.dat', 'rb')
    # memmap = mmap.mmap(memmap_fd.fileno(), 0, access=mmap.ACCESS_READ)

    def shutdown(sig=0, sframe=None):
        log.info('Worker shutting down')
        # memmap.close()
        # memmap_fd.close()
        sys.exit(0)

    def shutdown_ignore(sig=0, sframe=None):
        pass

    signal.signal(signal.SIGTERM, shutdown_ignore)
    signal.signal(signal.SIGINT, shutdown_ignore)

    log.info('Worker started. PID: {}. Parent PID: {}'.format(os.getpid(), os.getppid()))

    simulator = play.Simulator(simlulator_params['movements'], simlulator_params['max_steps'])

    while True:
        try:
            log.debug('Checking for tasks')
            task = task_queue.get(timeout=1)

            log.debug('Got task for individual {}'.format(task.id))
            individual = pickle_task_to_individual(task)

            log.info('Simulating individual {}'.format(individual.id))
            simulator._simulate_individual(individual, render=False)

            result_queue.put(SimulationResult(individual.id, individual.fitness))
            task_queue.task_done()

        except queue.Empty:
            log.debug('Task queue is empty')

        if shutdown_event.is_set():
            shutdown()
            return
